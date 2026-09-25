#include "PILOT_KWS.h"
#include "pilot_kws_model.h"

#include <Arduino.h>
#include <MicroTFLite.h>
#include <math.h>
#include <string.h>

// ============================================================
// PILOT KWS - ESP32
// Librosa-compatible MFCC front end + INT8 MicroTFLite model
//
// Model input : [1, 13, 98, 1]
// Classes     : 0 Background, 1 Negative, 2 PILOT
// Audio       : 16 kHz, mono, 15720 samples
//
// IMPORTANT:
// - This version uses a 48 KB tensor arena because the model
//   requires more than ~40 KB.
// - The large audio buffer is NOT static/global. It is allocated
//   only while processAudio() is running, then released.
// - FFT input and output are kept separate.
// ============================================================

namespace {

// ---------------- Audio / MFCC ----------------

constexpr int SAMPLE_RATE   = 16000;
constexpr int N_MFCC        = 13;
constexpr int N_MELS        = 40;
constexpr int N_FFT         = 400;
constexpr int HOP_LENGTH    = 160;
constexpr int TARGET_FRAMES = 98;
constexpr int AUDIO_SAMPLES = 15720;
constexpr int FFT_BINS      = N_FFT / 2 + 1;

// Librosa defaults for this training pipeline:
// window='hann', center=True, pad_mode='constant',
// power=2, htk=False, norm='slaney', fmin=0, fmax=sr/2.
constexpr float FMIN = 0.0f;
constexpr float FMAX = 8000.0f;
constexpr float LOG_FLOOR = 1.0e-10f;
constexpr float TOP_DB = 80.0f;

// Model needs about 40.8 KB at runtime. 48 KB is the smallest
// arena previously found to be usable in this project.
constexpr size_t TENSOR_ARENA_SIZE = 48 * 1024;

// Keep arena static so it is not placed on the stack.
alignas(16) static uint8_t tensorArena[TENSOR_ARENA_SIZE];

// FFT / MFCC work buffers.
// These are intentionally modest and reused.
static float* fftInput = nullptr;
static float* fftReal = nullptr;
static float* fftImag = nullptr;
static float* powerSpectrum = nullptr;
static float* melEnergies = nullptr;
static float* mfccFrame = nullptr;
static float* modelInput = nullptr;

static float* melBank = nullptr;
static float* dctCos = nullptr;
static float* hannWindow = nullptr;

static bool dspInitialized = false;
static bool modelInitialized = false;

static float pilotConfidence = 0.0f;
static float backgroundConfidence = 0.0f;
static float negativeConfidence = 0.0f;

static void buildWindow()
{
    // Librosa/scipy periodic Hann:
    // 0.5 - 0.5*cos(2*pi*n/N)
    for (int n = 0; n < N_FFT; ++n) {
        hannWindow[n] =
            0.5f - 0.5f * cosf(
                2.0f * PI * (float)n / (float)N_FFT
            );
    }
}

static float hzToMelSlaney(float hz)
{
    // Librosa's default Slaney mel conversion.
    constexpr float f_sp = 200.0f / 3.0f;
    constexpr float minLogHz = 1000.0f;
    constexpr float minLogMel = minLogHz / f_sp;
    constexpr float logstep = 0.06875177742094912f;

    if (hz < minLogHz) {
        return hz / f_sp;
    }

    return minLogMel +
           logf(hz / minLogHz) / logstep;
}

static float melToHzSlaney(float mel)
{
    constexpr float f_sp = 200.0f / 3.0f;
    constexpr float minLogHz = 1000.0f;
    constexpr float minLogMel = minLogHz / f_sp;
    constexpr float logstep = 0.06875177742094912f;

    if (mel < minLogMel) {
        return mel * f_sp;
    }

    return minLogHz *
           expf(logstep * (mel - minLogMel));
}

static void buildMelBank()
{
    const float melMin = hzToMelSlaney(FMIN);
    const float melMax = hzToMelSlaney(FMAX);

    float points[N_MELS + 2];

    for (int i = 0; i < N_MELS + 2; ++i) {
        float mel =
            melMin +
            (melMax - melMin) *
            (float)i / (float)(N_MELS + 1);

        points[i] = melToHzSlaney(mel);
    }

    for (int m = 0; m < N_MELS; ++m) {

        float left   = points[m];
        float center = points[m + 1];
        float right  = points[m + 2];

        // Librosa's Slaney normalization.
        float enorm = 2.0f / (right - left);

        for (int k = 0; k < FFT_BINS; ++k) {

            float freq =
                (float)k *
                (float)SAMPLE_RATE /
                (float)N_FFT;

            float weight = 0.0f;

            if (freq > left && freq < center) {
                weight =
                    (freq - left) /
                    (center - left);
            }
            else if (freq >= center && freq < right) {
                weight =
                    (right - freq) /
                    (right - center);
            }
            else if (freq == center) {
                weight = 1.0f;
            }

            melBank[m * FFT_BINS + k] =
                weight * enorm;
        }
    }
}

static void buildDCT()
{
    // scipy.fftpack.dct(type=2, norm='ortho')
    //
    // DCT-II:
    // C[k] = alpha(k) * sum_n x[n] *
    //        cos(pi/N * (n+0.5)*k)
    for (int k = 0; k < N_MFCC; ++k) {

        const float alpha =
            (k == 0)
            ? sqrtf(1.0f / (float)N_MELS)
            : sqrtf(2.0f / (float)N_MELS);

        for (int n = 0; n < N_MELS; ++n) {

            dctCos[k * N_MELS + n] =
                alpha *
                cosf(
                    PI *
                    (float)k *
                    ((float)n + 0.5f) /
                    (float)N_MELS
                );
        }
    }
}

static bool initializeDSP()
{
    if (dspInitialized) {
        return true;
    }

    // Allocate DSP working memory on the heap instead of .dram0.bss.
    // Total is roughly 50 KB and is allocated at runtime.
    fftInput       = (float*)malloc(N_FFT * sizeof(float));
    fftReal        = (float*)malloc(FFT_BINS * sizeof(float));
    fftImag        = (float*)malloc(FFT_BINS * sizeof(float));
    powerSpectrum  = (float*)malloc(FFT_BINS * sizeof(float));
    melEnergies    = (float*)malloc(N_MELS * sizeof(float));
    mfccFrame      = (float*)malloc(N_MFCC * sizeof(float));
    modelInput     = (float*)malloc(N_MFCC * TARGET_FRAMES * sizeof(float));
    melBank        = (float*)malloc(N_MELS * FFT_BINS * sizeof(float));
    dctCos         = (float*)malloc(N_MFCC * N_MELS * sizeof(float));
    hannWindow     = (float*)malloc(N_FFT * sizeof(float));

    if (!fftInput || !fftReal || !fftImag ||
        !powerSpectrum || !melEnergies || !mfccFrame ||
        !modelInput || !melBank || !dctCos || !hannWindow) {

        Serial.println("ERROR: DSP heap allocation FAILED.");

        free(fftInput);
        free(fftReal);
        free(fftImag);
        free(powerSpectrum);
        free(melEnergies);
        free(mfccFrame);
        free(modelInput);
        free(melBank);
        free(dctCos);
        free(hannWindow);

        fftInput = nullptr;
        fftReal = nullptr;
        fftImag = nullptr;
        powerSpectrum = nullptr;
        melEnergies = nullptr;
        mfccFrame = nullptr;
        modelInput = nullptr;
        melBank = nullptr;
        dctCos = nullptr;
        hannWindow = nullptr;

        return false;
    }

    buildWindow();
    buildMelBank();
    buildDCT();

    dspInitialized = true;
    return true;
}

static void calculateDFT400(const float* input)
{
    // True N=400 DFT.
    //
    // This intentionally does NOT use a 512-point FFT.
    // The Python training pipeline uses n_fft=400.
    //
    // Input and output are separate so the input frame can
    // never be overwritten while calculating the DFT.

    for (int k = 0; k < FFT_BINS; ++k) {

        float real = 0.0f;
        float imag = 0.0f;

        float angle =
            -2.0f * PI *
            (float)k /
            (float)N_FFT;

        float cosStep = cosf(angle);
        float sinStep = sinf(angle);

        float c = 1.0f;
        float s = 0.0f;

        for (int n = 0; n < N_FFT; ++n) {

            const float x = input[n];

            real += x * c;
            imag += x * s;

            float nextC =
                c * cosStep - s * sinStep;

            float nextS =
                s * cosStep + c * sinStep;

            c = nextC;
            s = nextS;
        }

        fftReal[k] = real;
        fftImag[k] = imag;
    }
}

static void computePowerSpectrum()
{
    for (int k = 0; k < FFT_BINS; ++k) {

        float re = fftReal[k];
        float im = fftImag[k];

        // Librosa uses power=2.0.
        powerSpectrum[k] =
            re * re + im * im;
    }
}

static void computeMFCCFrame(
    const float* audio,
    int frameIndex,
    float* output
)
{
    // Librosa center=True with n_fft=400:
    //
    // frame start = frameIndex * 160 - 200
    //
    // Samples outside [0, len) are zero padded.
    const int start =
        frameIndex * HOP_LENGTH -
        (N_FFT / 2);

    for (int n = 0; n < N_FFT; ++n) {

        int sampleIndex = start + n;

        float sample = 0.0f;

        if (
            sampleIndex >= 0 &&
            sampleIndex < AUDIO_SAMPLES
        ) {
            sample = audio[sampleIndex];
        }

        fftInput[n] =
            sample * hannWindow[n];
    }

    calculateDFT400(fftInput);
    computePowerSpectrum();

    // Mel filterbank.
    for (int m = 0; m < N_MELS; ++m) {

        float energy = 0.0f;

        for (int k = 0; k < FFT_BINS; ++k) {
            energy +=
                powerSpectrum[k] *
                melBank[m * FFT_BINS + k];
        }

        if (energy < LOG_FLOOR) {
            energy = LOG_FLOOR;
        }

        melEnergies[m] = energy;
    }

    // Librosa:
    // power_to_db(S, ref=1.0, amin=1e-10, top_db=80)
    //
    // Since ref=1, 10*log10(S) is used.
    float maxDb = -INFINITY;

    for (int m = 0; m < N_MELS; ++m) {

        float db =
            10.0f *
            log10f(
                fmaxf(
                    melEnergies[m],
                    LOG_FLOOR
                )
            );

        melEnergies[m] = db;

        if (db > maxDb) {
            maxDb = db;
        }
    }

    const float floorDb =
        maxDb - TOP_DB;

    for (int m = 0; m < N_MELS; ++m) {

        if (melEnergies[m] < floorDb) {
            melEnergies[m] = floorDb;
        }
    }

    // DCT-II -> first 13 coefficients.
    for (int k = 0; k < N_MFCC; ++k) {

        float sum = 0.0f;

        for (int m = 0; m < N_MELS; ++m) {
            sum +=
                dctCos[k * N_MELS + m] *
                melEnergies[m];
        }

        output[k] = sum;
    }
}

static bool buildMFCC(
    const int16_t* audio,
    size_t samples
)
{
    if (audio == nullptr) {
        return false;
    }

    if (samples < AUDIO_SAMPLES) {
        return false;
    }

    // Convert int16 PCM to the same approximate [-1,1]
    // floating-point range used by librosa.load().
    //
    // We deliberately use only the first 15720 samples.
    float* audioFloat =
        (float*)malloc(
            AUDIO_SAMPLES * sizeof(float)
        );

    if (audioFloat == nullptr) {
        Serial.println(
            "ERROR: Could not allocate MFCC audio buffer."
        );
        return false;
    }

    for (int i = 0; i < AUDIO_SAMPLES; ++i) {
        audioFloat[i] =
            (float)audio[i] / 32768.0f;
    }

    // Generate exactly 98 frames.
    //
    // IMPORTANT:
    // modelInput is coefficient-first:
    // [mfcc0 frame0..97,
    //  mfcc1 frame0..97, ...]
    //
    // This matches the Python X shape:
    // (samples, 13, 98)
    for (int frame = 0;
         frame < TARGET_FRAMES;
         ++frame)
    {
        computeMFCCFrame(
            audioFloat,
            frame,
            mfccFrame
        );

        for (int coefficient = 0;
             coefficient < N_MFCC;
             ++coefficient)
        {
            modelInput[
                coefficient * TARGET_FRAMES +
                frame
            ] =
                mfccFrame[coefficient];
        }
    }

    free(audioFloat);
    return true;
}

} // namespace


// ============================================================
//                    PUBLIC CLASS
// ============================================================

PILOT_KWS_Class PILOT_KWS;


// ============================================================
//                         BEGIN
// ============================================================

bool PILOT_KWS_Class::begin()
{
    pilotConfidence = 0.0f;
    backgroundConfidence = 0.0f;
    negativeConfidence = 0.0f;

    if (!initializeDSP()) {
        modelInitialized = false;
        return false;
    }

    Serial.println();
    Serial.println("PILOT KWS");
    Serial.println("--------------------");

    Serial.print("Model size: ");
    Serial.print(pilot_kws_tflite_len);
    Serial.println(" bytes");

    // MicroTFLite uses the model array directly.
    if (!ModelInit(
            pilot_kws_tflite,
            tensorArena,
            TENSOR_ARENA_SIZE
        ))
    {
        Serial.println(
            "ERROR: TFLite interpreter initialization FAILED!"
        );

        modelInitialized = false;
        return false;
    }

    modelInitialized = true;

    Serial.println(
        "TFLite interpreter initialized!"
    );

    Serial.print("Tensor arena: ");
    Serial.print(TENSOR_ARENA_SIZE / 1024);
    Serial.println(" KB");

    return true;
}


// ============================================================
//                    PROCESS AUDIO
// ============================================================

bool PILOT_KWS_Class::processAudio(
    const int16_t* audio,
    size_t samples
)
{
    if (!modelInitialized) {
        Serial.println(
            "ERROR: PILOT KWS model is not initialized."
        );
        return false;
    }

    if (audio == nullptr) {
        Serial.println(
            "ERROR: Audio pointer is NULL."
        );
        return false;
    }

    if (samples < AUDIO_SAMPLES) {
        Serial.print(
            "ERROR: Not enough audio samples. Got "
        );
        Serial.print(samples);
        Serial.print(", need ");
        Serial.println(AUDIO_SAMPLES);
        return false;
    }

    if (!buildMFCC(audio, samples)) {
        Serial.println(
            "ERROR: MFCC generation failed."
        );
        return false;
    }

    // --------------------------------------------------------
    // Send all 1274 input values individually.
    //
    // MicroTFLite 1.0.4 API:
    // ModelSetInput(float inputValue, int index)
    // --------------------------------------------------------

    for (int i = 0;
         i < N_MFCC * TARGET_FRAMES;
         ++i)
    {
        if (!ModelSetInput(
                modelInput[i],
                i
            ))
        {
            Serial.print(
                "ERROR: ModelSetInput failed at index "
            );
            Serial.println(i);

            return false;
        }
    }

    // --------------------------------------------------------
    // Run inference
    // --------------------------------------------------------

    if (!ModelRunInference()) {

        Serial.println(
            "ERROR: ModelRunInference() FAILED."
        );

        return false;
    }

    // --------------------------------------------------------
    // Read output probabilities
    // --------------------------------------------------------

    backgroundConfidence =
        ModelGetOutput(0);

    negativeConfidence =
        ModelGetOutput(1);

    pilotConfidence =
        ModelGetOutput(2);

    // Safety clamp.
    backgroundConfidence =
        constrain(
            backgroundConfidence,
            0.0f,
            1.0f
        );

    negativeConfidence =
        constrain(
            negativeConfidence,
            0.0f,
            1.0f
        );

    pilotConfidence =
        constrain(
            pilotConfidence,
            0.0f,
            1.0f
        );

    return true;
}


// ============================================================
//                    DETECTION
// ============================================================

bool PILOT_KWS_Class::isPilotDetected()
{
    // Keep threshold deliberately low for diagnostics.
    // Once the MFCC/model pipeline is verified, this can be
    // increased to reduce false activations.
    constexpr float DETECTION_THRESHOLD = 0.03f;

    return
        pilotConfidence >=
        DETECTION_THRESHOLD;
}


// ============================================================
//                       GETTERS
// ============================================================

float PILOT_KWS_Class::getPilotConfidence()
{
    return pilotConfidence;
}

float PILOT_KWS_Class::getBackgroundConfidence()
{
    return backgroundConfidence;
}

float PILOT_KWS_Class::getNegativeConfidence()
{
    return negativeConfidence;
}
