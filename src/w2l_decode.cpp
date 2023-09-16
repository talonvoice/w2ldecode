#include "w2l_decode.h"
#include "w2l_decode_backend.cpp"

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

using namespace w2l;

extern "C" {

w2l_decode_options w2l_decode_defaults {
    nullptr, // transitions
    0, // transitions_size
    "none", // criterion

    // decode options
    500, // beamsize
    100, // beamsizetoken
    25,  // beamthresh
    1.0, // lmweight
    1.0, // wordscore
    -INFINITY, // unkweight
    false, // logadd
    0.0,   // silweight

    // DFA decode options
    1.0,   // command_score
    0.55,  // rejection_threshold
    8,     // rejection_window_frames
    false, // debug
};

DLLEXPORT w2l_decoder *w2l_decoder_new(const char *tokens, const char *kenlm_model_path, const char *lexicon_path, const w2l_decode_options *opts) {
    // TODO: what other config? beam size? smearing? lm weight?
    auto decoder = new PublicDecoder(tokens, kenlm_model_path, lexicon_path, opts);
    return reinterpret_cast<w2l_decoder *>(decoder);
}

DLLEXPORT bool w2l_decoder_load_trie(w2l_decoder *c_decoder, const char *trie_path) {
    auto decoder = reinterpret_cast<PublicDecoder *>(c_decoder);
    return decoder->loadTrie(trie_path);
}

DLLEXPORT bool w2l_decoder_make_trie(w2l_decoder *c_decoder, const char *trie_path) {
    auto decoder = reinterpret_cast<PublicDecoder *>(c_decoder);
    return decoder->makeTrie(trie_path);
}

DLLEXPORT char *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission) {
    return reinterpret_cast<PublicDecoder *>(decoder)->decode(emission);
}

DLLEXPORT void w2l_decoder_free(w2l_decoder *decoder) {
    if (decoder)
        delete reinterpret_cast<PublicDecoder *>(decoder);
}

DLLEXPORT char *w2l_decoder_dfa(w2l_decoder *decoder, w2l_emission *emission, uint8_t *dfa, size_t dfa_size) {
    return reinterpret_cast<PublicDecoder *>(decoder)->decodeDFA(emission, dfa, dfa_size);
}

DLLEXPORT struct w2l_decoder_result *w2l_decoder_dfa_paths(w2l_decoder *decoder, w2l_emission *emission, uint8_t *dfa, size_t dfa_size) {
    return reinterpret_cast<PublicDecoder *>(decoder)->decodeDFAPaths(emission, dfa, dfa_size);
}

DLLEXPORT char *w2l_decoder_greedy(w2l_decoder *c_decoder, w2l_emission *emission) {
    auto decoder = reinterpret_cast<PublicDecoder *>(c_decoder);
    std::vector<int> path(emission->n_frames);
    decoder->decodeGreedy(emission, &path[0]);

    struct ResultInfo info;
    decoder->tokensDedupToWords(path, 0, path.size() - 1, info.words);
    auto text = info.joinWords();
    return strdup(text.c_str());
}

} // extern "C"
