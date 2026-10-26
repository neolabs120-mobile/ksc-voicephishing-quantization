// llama.cpp b9010의 공개 C API를 이용하는 로컬 평가기.
// 입력 JSONL: prompt, input_ids(HF 기준), label_ids, 선택적 generate_tokens.
// HTTP/채팅 서버의 템플릿·샘플링 개입 없이 전체 어휘 첫 토큰 argmax를 평가한다.
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text) {
    // prompt에는 HF에서 렌더링한 <bos>가 이미 들어 있다. 다시 붙이면 학습 때의 토큰열과 달라진다.
    // 먼저 필요한 배열 크기를 질의한 뒤 같은 옵션으로 실제 토큰 ID를 얻는다.
    int count = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
    std::vector<llama_token> ids(count);
    int actual = llama_tokenize(vocab, text.data(), text.size(), ids.data(), count, false, true);
    if (actual < 0) throw std::runtime_error("tokenize failed");
    ids.resize(actual);
    return ids;
}

// 전체 어휘의 최대 점수를 고른다. 0/1 외 토큰도 나올 수 있어 형식 실패를 따로 집계할 수 있다.
static int argmax(const float *logits, int size) {
    for (int i = 0; i < size; ++i)
        if (!std::isfinite(logits[i])) throw std::runtime_error("NaN/Inf logits");
    return std::max_element(logits, logits + size) - logits;
}

// 고정 토큰 입력을 평가한다. 기존 32K 명령과 RAG 실험의 명시적 문맥 한도를 함께 지원한다.
int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: gguf_eval MODEL INPUT_JSONL OUTPUT_JSONL GPU_LAYERS KV_TYPE(f32|f16) [CONTEXT_LIMIT]\n";
        return 2;
    }
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    try {
        if (std::ifstream(argv[3]).good()) throw std::runtime_error("output already exists");
        std::ifstream input(argv[2]);
        if (!input) throw std::runtime_error("input missing");
        std::ofstream output(argv[3]);
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = std::stoi(argv[4]);
        ggml_backend_dev_t no_devices[] = {nullptr};
        if (mp.n_gpu_layers == 0) mp.devices = no_devices;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");

        // 기존 오프라인 평가는 32K 기본값을 유지한다. RAG의 8K 한도는 선택 인자로 고정한다.
        auto cp = llama_context_default_params();
        cp.n_ctx = argc == 7 ? std::stoi(argv[6]) : 32768;
        if (cp.n_ctx < 512 || cp.n_ctx > 32768) throw std::runtime_error("invalid context limit");
        cp.n_batch = cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4;
        // 변환 검증에서는 KV도 FP32. FP16 KV는 별도 실행 조건으로 기록한다.
        std::string kv = argv[5];
        if (kv != "f32" && kv != "f16") throw std::runtime_error("invalid KV type");
        cp.type_k = cp.type_v = kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = cp.op_offload = mp.n_gpu_layers != 0;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context init failed");
        auto vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        int count = 0;
        std::string line;
        while (std::getline(input, line)) {
            auto row = json::parse(line);
            auto ids = tokenize(vocab, row.at("prompt"));
            // 단순 문장 길이 비교가 아니라 HF에서 저장한 모든 토큰 ID와 일치해야 진행한다.
            auto expected = row.at("input_ids").get<std::vector<llama_token>>();
            if (ids != expected) throw std::runtime_error("HF/GGUF tokenizer mismatch: " + row.at("id").get<std::string>());
            const int generate = row.value("generate_tokens", 0);
            if (ids.empty() || ids.size() + generate > cp.n_ctx)
                throw std::runtime_error("context overflow; input is never truncated");
            // 표본마다 KV 캐시를 초기화해 앞 통화 내용이 다음 통화 판정에 섞이지 않게 한다.
            llama_memory_clear(llama_get_memory(ctx), true);
            auto start = Clock::now();
            // prefill: 입력 전체를 512토큰씩 처리한다. 판정은 마지막 입력 위치의 다음 토큰 점수다.
            for (size_t offset = 0; offset < ids.size(); offset += cp.n_batch) {
                auto batch = llama_batch_get_one(ids.data() + offset,
                    std::min<size_t>(cp.n_batch, ids.size() - offset));
                if (llama_decode(ctx, batch)) throw std::runtime_error("prompt decode failed");
            }
            const auto *logits = llama_get_logits_ith(ctx, -1);
            int next = argmax(logits, n_vocab);
            auto labels = row.at("label_ids").get<std::vector<int>>();
            double l0 = logits[labels.at(0)], l1 = logits[labels.at(1)];
            // conditional_p1은 0/1 점수만 재정규화한 상대값이다. 실제 피싱 확률로 보정된 값은 아니다.
            // 지수의 큰 양수 계산을 피해 overflow 없이 sigmoid를 계산한다.
            double delta = l1 - l0;
            double p1 = delta >= 0 ? 1 / (1 + std::exp(-delta)) : std::exp(delta) / (1 + std::exp(delta));
            json result = {{"id", row.at("id")}, {"split", row.at("split")},
                {"label", row.at("label")}, {"text_sha256", row.at("text_sha256")},
                {"tokens", ids.size()}, {"tokenizer_match", true}, {"next_token", next},
                {"logit_0", l0}, {"logit_1", l1}, {"conditional_p1", p1},
                {"prefill_ms", std::chrono::duration<double, std::milli>(Clock::now() - start).count()}};
            // 전체 어휘 argmax가 라벨 토큰이 아니면 null로 남긴다. 강제 0/1 선택은 Python에서 별도 집계한다.
            result["prediction"] = next == labels[0] ? json(0) : next == labels[1] ? json(1) : json(nullptr);

            // 설명 생성은 greedy, 반복 패널티 없음. 생성 토큰은 Python에서 원본 토크나이저로 복호화한다.
            std::vector<int> generated;
            bool stopped = false;
            for (int i = 0; i < generate; ++i) {
                if (llama_vocab_is_eog(vocab, next)) { stopped = true; break; }
                generated.push_back(next);
                llama_token token = next;
                auto batch = llama_batch_get_one(&token, 1);
                if (llama_decode(ctx, batch)) throw std::runtime_error("generation decode failed");
                next = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            }
            if (generate) {
                result["generated_ids"] = generated;
                result["finish_reason"] = stopped ? "stop" : "length";
            }
            // 한 건씩 flush하므로 중단 전 완료된 예측을 확인할 수 있다. 재개/무결성 검사는 Python이 담당한다.
            output << result.dump() << '\n' << std::flush;
            if (!output) throw std::runtime_error("output write failed");
            if (++count % 50 == 0) std::cerr << "[PROGRESS] " << count << "\n";
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        return 1;
    }
}
