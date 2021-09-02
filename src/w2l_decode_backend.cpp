#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <unordered_set>

#include "fl-derived/Transforms.h"
#include "fl-derived/Dictionary.h"
#include "fl-derived/DecoderUtils.h"
#include "fl-derived/LibraryUtils.h"
#include "fl-derived/Trie.h"
#include "fl-derived/WordUtils.h"

#include "fl-derived/ZeroLM.h"
#ifdef USE_KENLM
#include "fl-derived/KenLM.h"
#endif

// for viterbi path
#include "fl-derived/ViterbiPath.h"

using namespace w2l;

#include "w2l_decode.h"
#include "decode_core.cpp"

struct WordInfo {
    std::string word;
    int start, end;

    WordInfo(std::string _word, int _start, int _end) :
        word(_word), start(_start), end(_end) {
            ;
    }
};

struct ResultInfo {
    std::string text;
    double score;
    std::vector<struct WordInfo> words;

    std::string joinWords() {
        std::ostringstream ostr;
        if (!words.empty()) {
            ostr << words.front().word;
            for (size_t i = 1; i < words.size(); i++) {
                ostr << " " << words[i].word;
            }
        }
        return ostr.str();
    }
};

class GreedyDecoder {
public:
    static void decodeASG(w2l_emission *emission, float *transitions, int *path_out) {
        int T = emission->n_frames;
        int N = emission->n_tokens;
        float *emissions = emission->matrix;
        std::vector<uint8_t> workspace(w2l::cpu::ViterbiPath<float>::getWorkspaceSize(1, T, N));
        w2l::cpu::ViterbiPath<float>::compute(
            1, // B
            T,
            N,
            emissions,
            transitions,
            path_out,
            workspace.data());
    }

    static void decodeCTC(w2l_emission *emission, int *path_out) {
        int T = emission->n_frames;
        int N = emission->n_tokens;
        float *emissions = emission->matrix;
        // CTC viterbi is just argmax
        for (int t = 0; t < T; t++) {
            auto it = std::max_element(emissions, emissions + N);
            path_out[t] = std::distance(emissions, it);
            emissions += N;
        }
    }
};

DecoderOptions toW2lDecoderOptions(const w2l_decode_options &opts) {
    CriterionType criterionType;
    if (std::string(opts.criterion) == kCtcCriterion) {
        criterionType = CriterionType::CTC;
    } else if (std::string(opts.criterion) == kAsgCriterion) {
        criterionType = CriterionType::ASG;
    } else {
        std::cerr << "[Decoder] Invalid criterion type: " << opts.criterion << std::endl;
        abort();
    }
    return DecoderOptions(
                opts.beamsize,
                opts.beamsizetoken,
                opts.beamthresh,
                opts.lmweight,
                opts.wordscore, // lexiconscore
                opts.unkweight, // unkscore
                opts.silweight, // silscore
                0,              // eosscore
                opts.logadd,
                criterionType);
}

std::vector<std::string> loadWordList(const char *path) {
    std::vector<std::string> result;
    result.reserve(1000000);
    result.push_back("<unk>");

    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line)) {
        auto sep = std::min(line.find("\t"), line.find(" "));
        auto word = line.substr(0, sep);
        // handle duplicate words
        if (result.size() == 0 || word != result.back()) {
            result.push_back(word);
        }
    }

    return result;
}

static int getSilIdx(Dictionary &tokenDict) {
    if (tokenDict.contains(kSilToken)) {
        return tokenDict.getIndex(kSilToken);
    } else if (tokenDict.contains("_")) {
        return tokenDict.getIndex("_");
    } else if (tokenDict.contains("|")) {
        return tokenDict.getIndex("|");
    }
    return 0;
}

class PublicDecoder {
public:
    PublicDecoder(const char *tokens, const char *languageModelPath, const char *lexiconPath, const w2l_decode_options *opts) {
        this->lexiconPath = lexiconPath;
        this->setOptions(opts);
        auto tokenStream = std::istringstream(tokens);
        tokenDict = Dictionary(tokenStream);
        // TODO: ensure that resulting tokenDict.indexSize() > 0?
        if (tokenDict.indexSize() <= 0) {
            std::cerr << "[Decoder] tokenDict.indexSize() <= 0" << std::endl;
            abort();
        }
        if (decoderOpt.criterionType == CriterionType::CTC &&
                tokenDict.indexSize() > 0 &&
                tokenDict.getEntry(tokenDict.indexSize() - 1) != kBlankToken) {
            tokenDict.addEntry(kBlankToken);
        }

        globalTokens = &tokenDict;
        silIdx = getSilIdx(tokenDict);
        if (tokenDict.contains(kBlankToken)) {
            blankIdx = tokenDict.getIndex(kBlankToken);
        } else {
            blankIdx = -1;
        }
        if (decoderOpt.criterionType == CriterionType::ASG) {
            blankIdx = silIdx;
        }

        wordList = loadWordList(lexiconPath);
#ifdef USE_KENLM
        if (languageModelPath) {
            std::vector<std::string> lowercaseWordList;
            lowercaseWordList.reserve(wordList.size());
            for (std::string word : wordList) {
                std::transform(word.begin(), word.end(), word.begin(), [](char c){ return std::tolower(c); });
                lowercaseWordList.push_back(word);
            }
            lm = std::make_shared<KenLM>(languageModelPath, lowercaseWordList);
        }
#endif
        if (! lm) {
            lm = std::make_shared<ZeroLM>();
        }
    }
    ~PublicDecoder() {}

    // FLATTRIE HEADER:
    //  0: 'FLAT' (magic)
    //  4: 4-byte version (2)
    //  8: 32-byte  src file hash slot (written as zeroes, up to the caller to fill/check it)
    // 40: 32-byte trie file hash slot (written as zeroes, up to the caller to fill/check it)
    // 72: 8-byte data size
    // 80: data...
    bool loadTrie(const char *triePath) {
        // Load the trie
        std::ifstream f(triePath, std::ios::binary | std::ios::in);
        if (!f) {
            return false;
        }
        char     magic[4];
        char     srcHash[32];
        char     trieHash[32];
        uint32_t version  = 0;
        size_t   byteSize = 0;

        f.read(         magic,    4);
        f.read((char *)&version,  sizeof(version));
        f.read(         srcHash,  sizeof(srcHash));
        f.read(         trieHash, sizeof(trieHash));
        f.read((char *)&byteSize, sizeof(byteSize));
        if (memcmp(magic, "FLAT", 4) != 0) {
            return false;
        }
        if (version != 2) {
            return false;
        }
        if ((byteSize % 4) != 0) {
            return false;
        }
        flatTrie = std::make_shared<FlatTrie>();
        flatTrie->storage.resize(byteSize / 4);
        f.read(reinterpret_cast<char *>(flatTrie->storage.data()), byteSize);

        // the root maxScore should be 0 during search and it's more convenient to set here
        const_cast<FlatTrieNode *>(flatTrie->getRoot())->maxScore = 0;
        return f.good();
    }

    bool makeTrie(const char *triePath) {
        if (!lm) {
            return false;
        }
        auto lexicon = loadWords(lexiconPath, -1);
        Dictionary wordDict;
        for (const auto& it : wordList) {
            wordDict.addEntry(it);
        }
        wordDict.setDefaultIndex(wordDict.getIndex(kUnkToken));

        // taken from Decode.cpp
        // Build Trie
        std::vector<float> wordScores(wordDict.indexSize());
        Trie trie(tokenDict.indexSize(), silIdx);
        auto startState = lm->start(false);
        for (auto& it : lexicon) {
            const std::string& word = it.first;
            int usrIdx = wordDict.getIndex(word);
            float score = -1;
            // if (FLAGS_decodertype == "wrd")
            if (true) {
                LMStatePtr dummyState;
                std::tie(dummyState, score) = lm->score(startState, usrIdx);
            }
            for (auto& tokens : it.second) {
                auto tokensTensor = tkn2Idx(tokens, tokenDict, false /* replabel */ );
                trie.insert(tokensTensor, usrIdx, score);
                wordScores[usrIdx] = score;
            }
        }

        // Smearing
        // TODO: smear mode argument?
        SmearingMode smear_mode = SmearingMode::MAX;
        /*
        SmearingMode smear_mode = SmearingMode::NONE;
        if (FLAGS_smearing == "logadd") {
            smear_mode = SmearingMode::LOGADD;
        } else if (FLAGS_smearing == "max") {
            smear_mode = SmearingMode::MAX;
        } else if (FLAGS_smearing != "none") {
            LOG(FATAL) << "[Decoder] Invalid smearing mode: " << FLAGS_smearing;
        }
        */
        trie.smear(smear_mode);

        auto flatTrie = toFlatTrie(trie.getRoot(), wordScores);
        std::ofstream out(triePath, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!out.is_open())
            return false;

        // header is described above the loadTrie() method
        const char *magic = "FLAT";
        uint32_t version = 2;
        char zeroHash[32];
        memset(zeroHash, 0, 32);
        uint64_t byteSize = 4 * flatTrie.storage.size();

        out.write(          magic,    4);
        out.write((char *) &version,  sizeof(version));
        out.write(          zeroHash, sizeof(zeroHash));
        out.write(          zeroHash, sizeof(zeroHash));
        out.write((char *) &byteSize, sizeof(byteSize));
        out.write((char *) (flatTrie.storage.data()), byteSize);
        out.close();
        return out.good();
    }

    void setOptions(const w2l_decode_options *opts) {
        // safely retain external opts by copying transitions array and criterion string
        this->opts = *opts;
        this->opts.transitions = nullptr;
        this->transitions.resize(opts->transitions_len);
        if (opts->transitions != nullptr && opts->transitions_len > 0) {
            std::copy(opts->transitions, opts->transitions + opts->transitions_len, this->transitions.begin());
        }
        if (opts->criterion == std::string("asg")) {
            this->opts.criterion = "asg";
        } else if (opts->criterion == std::string("ctc")) {
            this->opts.criterion = "ctc";
        } else if (opts->criterion == std::string("s2s")) {
            this->opts.criterion = "s2s";
        } else {
            this->opts.criterion = "";
        }
        decoderOpt = toW2lDecoderOptions(*opts);
    }

    char *decode(w2l_emission *emission) {
        const char *dfa = "\x01\x01\x00\xfe\xff\x00\x00\x00\x00";
        return decodeDFA(emission, (w2l_dfa_node *)dfa, sizeof(dfa));
    }

    void decodeGreedy(w2l_emission *emission, int *path) {
        if (decoderOpt.criterionType == CriterionType::CTC) {
            GreedyDecoder::decodeCTC(emission, path);
        } else if (decoderOpt.criterionType == CriterionType::ASG) {
            int64_t tokens_squared = emission->n_tokens * emission->n_tokens;
            if (transitions.size() != tokens_squared) {
                std::cerr << "[Decoder] transitions.size() {" << transitions.size() << "} != emission->n_tokens ** 2 {" << tokens_squared << "}" << std::endl;
                abort();
            }
            GreedyDecoder::decodeASG(emission, &transitions[0], path);
        } else {
            std::cerr << "[Decoder] Unknown criterion enum: " << (int)decoderOpt.criterionType << std::endl;
            abort();
        }
    }

    char *decodeDFA(w2l_emission *emission, w2l_dfa_node *dfa, size_t dfa_size);
    w2l_decoder_result *decodeDFAPaths(w2l_emission *emission, w2l_dfa_node *dfa, size_t dfa_size);
    w2l_decoder_result *viterbiResult(const std::string &text, double viterbiScore);

    std::string tokensToString(const std::vector<int> &tokens, int from, int to) {
        std::string out;
        for (int i = from; i < to; ++i) {
            out.append(tokenDict.getEntry(tokens[i]));
        }
        return out;
    };

    void tokensDedupToWords(const std::vector<int> &tokens, int from, int to,
            std::vector<struct WordInfo> &wordsOut) {
        std::ostringstream oword;
        int tok = -1;
        int lastSilence = from - 1;
        for (int i = from; i < to; ++i) {
            if (tok == tokens[i]) {
                if (tok == silIdx)
                    lastSilence = i;
                continue;
            }
            tok = tokens[i];
            if (tok >= 0 && tok != blankIdx) {
                std::string s = tokenDict.getEntry(tok);
                if (!s.empty() && (s[0] == '_' || s[0] == '|')) {
                    if (oword.tellp() > 0) {
                        // FIXME: use last non blank index here?
                        // or for a run of blanks, use the first blank?
                        int start = lastSilence + 1;
                        int end = i;
                        wordsOut.emplace_back(oword.str(), start, end);
                        oword = std::ostringstream();
                    }
                    s = s.substr(1);
                    lastSilence = i;
                }
                oword << s;
            }
        }
        if (oword.tellp() > 0) {
            int start = lastSilence + 1;
            int end = to;
            wordsOut.emplace_back(oword.str(), start, end);
        }
    };

    LMPtr lm;
    FlatTriePtr flatTrie;
    std::string lexiconPath;
    std::vector<std::string> wordList;
    Dictionary tokenDict;
    DecoderOptions decoderOpt;
    int silIdx;
    int blankIdx;
    int unkLabel = 0;

    std::vector<float> transitions;
    w2l_decode_options opts = {};
};

namespace DFALM {

// flags on dfa nodes
enum {
    FLAG_NONE    = 0,
    FLAG_TERM    = 1,
};

// special token values on dfa edges
enum {
    TOKEN_LMWORD      = 0xffff,
    TOKEN_LMWORD_CTX  = 0xfffe,
    TOKEN_SKIP        = 0xfffd,
};

struct LM {
    LMPtr scorer;
    LMStatePtr lmStart;
    FlatTriePtr trie;
    const w2l_dfa_node *dfa;
    int silToken;
    int firstCommandLabel = 0;

    const w2l_dfa_node *get(const w2l_dfa_node *base, const int32_t idx) const {
        return reinterpret_cast<const w2l_dfa_node *>(reinterpret_cast<const uint8_t *>(base) + idx);
    }

    float commandScore = 1.5;
};

struct State {
    // pos in the grammar, never null
    const w2l_dfa_node *grammarLex = nullptr;
    // pos in trie, only set while decoding a lexicon word
    const FlatTrieNode *dictLex = nullptr;
    // LM state, preserved even when dictLex goes nullptr
    LMStatePtr lmState = nullptr;
    // whether the last edge in the grammar was a silToken.
    // could be optimized away
    bool wordEnd = false;

    // used for making an unordered_set of const State*
    struct Hash {
        const LM &unused;
        size_t operator()(const State *v) const {
            return std::hash<const void*>()(v->grammarLex) ^ std::hash<const void*>()(v->dictLex);
        }
    };

    struct Equality {
        const LM &lm_;
        int operator()(const State *v1, const State *v2) const {
            return v1->grammarLex == v2->grammarLex
                && v1->dictLex == v2->dictLex
                && (v1->lmState == v2->lmState
                    || (v1->lmState && v2->lmState && v1->lmState == v2->lmState));
        }
    };

    // Iterate over labels, calling fn with: the new State, the label index and the lm score
    template <typename Fn>
    void forLabels(const LM &lm, Fn&& fn) const {
        // in dictionary mode we may return positive labels for dictionary words
        if (dictLex) {
            const auto n = dictLex->nLabel;
            for (int i = 0; i < n; ++i) {
                int label = dictLex->label(i);
                float score = 0.0;
                State it;
                it.grammarLex = grammarLex;
                it.dictLex = nullptr;

                std::tie(it.lmState, score) = lm.scorer->score(lmState, label);
                auto zeroLM = dynamic_cast<ZeroLM *>(lm.scorer.get());
                if (zeroLM) {
                    score += dictLex->score(i);
                }
                fn(std::move(it), label, score);
            }
            return;
        }

        // command labels are offsets from lm.dfa, plus the firstCommandLabel value
        if (wordEnd) {
            fn(*this, lm.firstCommandLabel + (reinterpret_cast<const uint8_t*>(grammarLex) - reinterpret_cast<const uint8_t*>(lm.dfa)), lm.commandScore);
        }
    }

    // Call finish() on the lm, like for end-of-sentence scoring
    std::pair<State, float> finish(const LM &lm) const {
        bool bad = dictLex || !(grammarLex->flags & FLAG_TERM);
        return {*this, bad ? -1000000 : 0};
    }

    float maxWordScore() const {
        if (dictLex) {
            return dictLex->maxScore;
        }
        return 0; // could control whether the beam search gets scores before finishing commands
    }

    // Iterate over children of the state, calling fn with:
    // new State, new token index and whether the new state has children
    template <typename Fn>
    bool forChildren(int frame, std::unordered_set<int> &indices, const LM &lm, Fn&& fn) const {
        // If a dictionary word was started only consider its trie children.
        if (dictLex) {
            const auto n = dictLex->nChildren;
            for (int i = 0; i < n; ++i) {
                auto nlex = dictLex->child(i);
                if (indices.find(nlex->idx) != indices.end()) {
                    fn(State{grammarLex, nlex, lmState}, nlex->idx, nlex->nChildren > 0);
                }
            }
            return true;
        }

        // Otherwise look at the grammar dfa
        std::vector<const w2l_dfa_node *> queue = {grammarLex};
        while (queue.size() > 0) {
            auto dfaLex = queue.back();
            queue.pop_back();

            for (int i = 0; i < dfaLex->nEdges; ++i) {
                const auto &edge = dfaLex->edges[i];
                auto nlex = lm.get(dfaLex, edge.offset);

                // For dictionary edges start exploring the trie
                if (edge.token == TOKEN_LMWORD || edge.token == TOKEN_LMWORD_CTX) {
                    auto nextState = edge.token == TOKEN_LMWORD_CTX ? lmState : nullptr;
                    if (!nextState)
                        nextState = lm.lmStart;
                    auto dictRoot = lm.trie->getRoot();
                    const auto n = dictRoot->nChildren;
                    for (int i = 0; i < n; ++i) {
                        auto nDictLex = dictRoot->child(i);
                        if (indices.find(nDictLex->idx) != indices.end()) {
                            fn(State{nlex, nDictLex, nextState}, nDictLex->idx, nDictLex->nChildren > 0);
                        }
                    }
                } else if (edge.token == TOKEN_SKIP) {
                    // std::cout << "skip token, queueing up a new node with " << nlex->nEdges << " edges\n";
                    queue.push_back(nlex);
                } else if (indices.find(edge.token) != indices.end()) {
                    fn(State{nlex, nullptr, nullptr, edge.token == lm.silToken}, edge.token, true);
                }
            }
        }
        return true;
    }

    State &actualize() {
        return *this;
    }
};

} // namespace DFALM

using CombinedDecoder = SimpleDecoder<DFALM::LM, DFALM::State>;

// Score adjustment during beam search to reject beams early
// that diverge too much from the best emission-transmission score.
struct ViterbiDifferenceRejecter {
    std::vector<int> viterbiToks;

    // index i contains the emission-transmission score of up to windowMaxSize
    // previous frames of the viterbiTokens, see precomputeViterbiWindowScores.
    std::vector<float> viterbiWindowScores;

    int windowMaxSize;
    int silIdx;
    int blankIdx;
    float threshold;
    w2l_emission *emission;
    float *transitions;

    float extraNewTokenScore(int frame, const CombinedDecoder::DecoderState &prevState, int token) const {
        const int T = emission->n_frames;
        const int N = emission->n_tokens;
        auto refScore = viterbiWindowScores[frame];

        bool allSilence = (token == silIdx || token == blankIdx);
        int prevToken = token;
        auto thisState = &prevState;
        float thisScore = emission->matrix[frame * N + token];
        int thisWindow = 1;
        while (thisWindow < windowMaxSize && thisState && frame - thisWindow >= 0) {
            token = thisState->getToken();
            if (token != silIdx && token != blankIdx)
                allSilence = false;
            thisScore += emission->matrix[(frame - thisWindow) * N + token];
            if (transitions) {
                thisScore += transitions[prevToken * N + token];
            }
            ++thisWindow;
            prevToken = token;
            thisState = thisState->parent;
        }

        // rejecting based on non-full windows is too unstable, wait for full window
        if (thisWindow < windowMaxSize)
            return 0;

        if (thisScore / refScore < threshold) {
            return -100000;
        }

        // Only allow a full silence window if the viterbi tok in the middle is also silence
        int middleTok = viterbiToks[frame - windowMaxSize/2];
        if (allSilence && middleTok != silIdx && middleTok != blankIdx) {
            return -100000;
        }

        return 0;
    }

    void precomputeViterbiWindowScores(int segStart, const std::vector<int> &viterbiToks) {
        const int T = emission->n_frames;
        const int N = emission->n_tokens;
        float score = 0;
        for (int j = segStart; j < T; ++j) {
            score += emission->matrix[(j - segStart) * N + viterbiToks[j]];
            if (j != segStart && transitions)
                score += transitions[viterbiToks[j] * N + viterbiToks[j - 1]];
            viterbiWindowScores.push_back(score);
            if (j - segStart < windowMaxSize - 1)
                continue;
            auto r = j - (windowMaxSize - 1);
            score -= emission->matrix[(r - segStart) * N + viterbiToks[r]];
            if (r != segStart && transitions)
                score -= transitions[viterbiToks[r] * N + viterbiToks[r - 1]];
        }
    }
};

char *PublicDecoder::decodeDFA(w2l_emission *emission, w2l_dfa_node *dfa, size_t dfa_size) {
    w2l_decoder_result *result = decodeDFAPaths(emission, dfa, dfa_size);
    if (!result || result->n_paths < 1) {
        return nullptr;
    }
    char *s = strdup(result->paths[0]->text);
    free(result);
    return s;
}

static float getViterbiScore(w2l_emission *emission, std::vector<int> &tokens, std::vector<float> &transitions) {
    double viterbiScore = 0;
    int N = emission->n_tokens;
    for (int t = 0; t < emission->n_frames; t++) {
        int n = tokens[t];
        viterbiScore += emission->matrix[t * N + n];
        if (!transitions.empty() && t > 0) {
            viterbiScore += transitions[n * N + tokens[t - 1]];
        }
    }
    return viterbiScore;
}

w2l_decoder_result *PublicDecoder::viterbiResult(const std::string &text, double viterbiScore) {
    size_t size = offsetof(struct w2l_decoder_result, paths[0]);
    auto result = (struct w2l_decoder_result *)calloc(1, size + text.size() + 1);
    char *buf = (char *)&result->paths[0];
    memcpy(buf, text.c_str(), text.size());
    result->greedy_text = buf;
    result->greedy_score = viterbiScore;
    result->n_paths = 0;
    return result;
}

w2l_decoder_result *PublicDecoder::decodeDFAPaths(w2l_emission *emission, w2l_dfa_node *dfa, size_t dfa_size) {
    std::vector<int> viterbiToks(emission->n_frames);
    decodeGreedy(emission, &viterbiToks[0]);

    struct ResultInfo viterbiInfo;
    tokensDedupToWords(viterbiToks, 0, viterbiToks.size() - 1, viterbiInfo.words);
    auto viterbiText = viterbiInfo.joinWords();
    double viterbiScore = getViterbiScore(emission, viterbiToks, transitions);

    // Lets skip decoding if viterbi thinks it's all silence
    bool allSilence = true;
    for (auto t : viterbiToks) {
        if (t != silIdx && t != blankIdx) {
            allSilence = false;
            break;
        }
    }
    if (allSilence) {
        return viterbiResult(viterbiText, viterbiScore);
    }

    LMStatePtr start = lm ? lm->start(0) : nullptr;
    auto dfalm = DFALM::LM{lm, start, flatTrie, dfa, silIdx};
    dfalm.commandScore = opts.command_score;
    dfalm.firstCommandLabel = wordList.size();

    auto commandDecoder = CombinedDecoder{
        dfalm,
        silIdx,
        blankIdx,
        unkLabel,
        transitions};

    if (opts.debug) {
        float viterbiScore = 0;
        int N = emission->n_tokens;
        for (int t = 0; t < emission->n_frames; t++) {
            int n = viterbiToks[t];
            viterbiScore += emission->matrix[t * N + n];
            if (!transitions.empty() && t > 0) {
                viterbiScore += transitions[n * N + viterbiToks[t - 1]];
            }
        }
        auto s = tokensToString(viterbiToks, 0, viterbiToks.size());
        std::cerr << s << " " << viterbiScore << " (viterbi)" << std::endl << std::endl;
    }

    ViterbiDifferenceRejecter rejecter;
    rejecter.windowMaxSize = opts.rejection_window_frames;
    if (rejecter.windowMaxSize <= 0) {
        rejecter.windowMaxSize = 1;
    }
    rejecter.threshold = opts.rejection_threshold;
    rejecter.emission = emission;
    rejecter.silIdx = silIdx;
    rejecter.blankIdx = blankIdx;
    if (transitions.size() == 0) {
        rejecter.transitions = NULL;
    } else {
        rejecter.transitions = transitions.data();
    }
    rejecter.precomputeViterbiWindowScores(0, viterbiToks);
    rejecter.viterbiToks = viterbiToks;

    DFALM::State commandState;
    commandState.grammarLex = dfalm.dfa;

    // in the future we could stop the decode after one word instead of
    // decoding everything
    std::vector<CombinedDecoder::DecoderState> startStates;
    startStates.emplace_back(commandState, nullptr, 0.0, silIdx, -1);

    auto unfinishedBeams = [&]() {
        const auto parallelBeamsearch = false;
        if (!parallelBeamsearch)
            return commandDecoder.normalAll(decoderOpt, emission, startStates, rejecter);

        int nThreads = 4;
        int stepsPerFanout = 5;
        int threadBeamSize = decoderOpt.beamSize / nThreads;
        return commandDecoder.groupThreading(decoderOpt, emission, startStates, rejecter, nThreads, stepsPerFanout, threadBeamSize);
    }();

    // Finishing kills beams that end in the middle of a word, or
    // in a grammar state that isn't TERM
    std::vector<CombinedDecoder::DecoderState> beamEnds;
    beamSearchFinish(beamEnds, unfinishedBeams.hyp.back(), dfalm, decoderOpt);
    if (beamEnds.empty()) {
        return viterbiResult(viterbiText, viterbiScore);
    }

    auto results = _getAllHypothesis(beamEnds, unfinishedBeams.hyp.size());
    // We never take rejected beams.
    // TODO: return rejected beams, and make the caller handle it by checking score?
    if (beamEnds[0].score < -90000) {
        return viterbiResult(viterbiText, viterbiScore);
    }

    std::vector<struct ResultInfo> resultsInfo;
    for (auto &result : results) {
        struct ResultInfo info;
        info.score = result.score;
        int lastSilence = -1;
        for (int i = 0; i < result.words.size() - 1; ++i) {
            const auto label = result.words[i];
            if (label >= 0 && label < dfalm.firstCommandLabel) {
                info.words.emplace_back(wordList[label], lastSilence + 1, i);
            } else if (label >= dfalm.firstCommandLabel) {
                struct ResultInfo commandInfo;
                tokensDedupToWords(result.tokens, lastSilence + 1, i, commandInfo.words);
                for (auto &word : commandInfo.words) {
                    if (decoderOpt.criterionType == CriterionType::ASG) {
                        info.words.emplace_back("@" + word.word, word.start, word.end);
                    } else {
                        info.words.emplace_back(word.word, word.start, word.end);
                    }
                }
            }
            const auto token = result.tokens[i];
            if (token == silIdx)
                lastSilence = i;
        }
        // TODO: include the tokens in the result so the caller can print debug info instead of us?
        if (opts.debug) {
            auto decoderToks = result.tokens;
            auto s = tokensToString(decoderToks, 1, decoderToks.size() - 1);
            std::cerr << s << " " << result.score << std::endl;
        }
        info.text = info.joinWords();
        resultsInfo.push_back(std::move(info));
    }

    if (opts.debug) {
        auto &text = resultsInfo.front().text;
        if (text.empty()) {
            std::cerr << "  [reject]";
        } else {
            std::cerr << "  result: \"" << text << "\"";
        }
        std::cerr << std::endl << std::endl;
    }

    // prepare result object
    size_t base_size      = offsetof(struct w2l_decoder_result, paths[0]);
    size_t path_data_size = offsetof(struct w2l_decoder_path,   words[0]) * resultsInfo.size();
    size_t path_ptr_size  = sizeof(struct w2l_decoder_path *) * resultsInfo.size();
    int strpos = 0;

    // build strtab offsets and count words
    size_t word_count = 0;
    strpos += viterbiText.size() + 1;
    for (auto &info : resultsInfo) {
        strpos += info.text.size() + 1;
        for (auto &word : info.words) {
            strpos += word.word.size() + 1;
        }
        word_count += info.words.size();
    }
    size_t path_size = path_ptr_size + path_data_size;
    size_t word_size = sizeof(struct w2l_decoder_word) * word_count;

    // allocate and fill result
    auto result = (struct w2l_decoder_result *)calloc(1, base_size + path_size + word_size + strpos);
    uintptr_t resultpos = (uintptr_t)result;
    uintptr_t pathpos   = resultpos + base_size + path_ptr_size;
    char *strtab = (char *)(pathpos + path_data_size + word_size);

    strcpy(strtab, viterbiText.c_str());
    result->greedy_text = strtab;
    strtab += viterbiText.size() + 1;
    result->greedy_score = viterbiScore;
    result->n_paths = resultsInfo.size();

    double frameCount = emission->n_frames;
    size_t path_i = 0;
    for (auto &info : resultsInfo) {
        auto path = (struct w2l_decoder_path *)pathpos;
        pathpos += offsetof(struct w2l_decoder_path, words[0]);
        pathpos += sizeof(struct w2l_decoder_word) * info.words.size();
        result->paths[path_i++] = path;

        strcpy(strtab, info.text.c_str());
        path->text = strtab;
        strtab += info.text.size() + 1;

        path->score = info.score;
        path->n_words = info.words.size();

        size_t word_i = 0;
        for (auto &word : info.words) {
            auto pathWord = &path->words[word_i++];
            strcpy(strtab, word.word.c_str());
            pathWord->word = strtab;
            strtab += word.word.size() + 1;

            pathWord->start = (double)word.start / frameCount;
            pathWord->end   = (double)word.end   / frameCount;
        }
    }
    /*
    printf("result:\n");
    printf("  greedy_text  = %s\n", result->greedy_text);
    printf("  greedy_score = %f\n", result->greedy_score);
    printf("  n_paths      = %d\n", result->n_paths);
    for (int path_i = 0; path_i < result->n_paths; path_i++) {
        auto path = result->paths[path_i];
        printf("  path[%d]:\n", path_i);
        printf("    text    = %s\n", path->text);
        printf("    score   = %f\n", path->score);
        printf("    n_words = %d\n", path->n_words);
        for (int word_i = 0; word_i < path->n_words; word_i++) {
            auto word = &path->words[word_i];
            printf("    words[%d]:\n", word_i);
            printf("      word  = %s\n", word->word);
            printf("      start = %f\n", word->start);
            printf("      end   = %f\n", word->end);
        }
    }
    */
    return result;
}
