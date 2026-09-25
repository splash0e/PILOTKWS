#include <Arduino.h>
#include <driver/i2s.h>
#include <PILOT_KWS.h>

// ============================================================
//                    PILOT KWS - ESP32
//        INMP441 + Custom TensorFlow Lite Micro Model
// ============================================================

// ------------------------- I2S -------------------------------

#define I2S_PORT I2S_NUM_0

#define I2S_BCLK 26
#define I2S_WS   25
#define I2S_DOUT 33

// ------------------------- LEDs ------------------------------

#define RED_LED  2
#define BLUE_LED 4

// ------------------------- Audio -----------------------------

#define SAMPLE_RATE 16000

// 98 MFCC frames × 160 hop + 400 FFT requirement
#define AUDIO_SAMPLES 15720

int16_t *audioBuffer = nullptr;

// Blue LED activation timer
unsigned long blueLedUntil = 0;


// ============================================================
//                     I2S SETUP
// ============================================================

void setupI2S()
{
    Serial.println();
    Serial.println("Initializing INMP441...");

    i2s_config_t i2s_config = {};

    i2s_config.mode =
        (i2s_mode_t)(
            I2S_MODE_MASTER |
            I2S_MODE_RX
        );

    i2s_config.sample_rate = SAMPLE_RATE;

    // INMP441 outputs 24-bit audio inside 32-bit I2S words
    i2s_config.bits_per_sample =
        I2S_BITS_PER_SAMPLE_32BIT;

    i2s_config.channel_format =
        I2S_CHANNEL_FMT_ONLY_LEFT;

    i2s_config.communication_format =
        I2S_COMM_FORMAT_STAND_I2S;

    i2s_config.intr_alloc_flags =
        ESP_INTR_FLAG_LEVEL1;

    i2s_config.dma_buf_count = 8;
    i2s_config.dma_buf_len = 256;

    i2s_config.use_apll = false;
    i2s_config.tx_desc_auto_clear = false;
    i2s_config.fixed_mclk = 0;

    esp_err_t result = i2s_driver_install(
        I2S_PORT,
        &i2s_config,
        0,
        NULL
    );

    if (result != ESP_OK)
    {
        Serial.print(
            "I2S driver installation FAILED: "
        );
        Serial.println(result);

        while (true)
        {
            digitalWrite(RED_LED, LOW);
            delay(500);

            digitalWrite(RED_LED, HIGH);
            delay(500);
        }
    }


    // ========================================================
    // IMPORTANT:
    // Initialize the entire structure.
    // This prevents the MCLK garbage-value problem.
    // ========================================================

    i2s_pin_config_t pin_config = {};

    pin_config.mck_io_num =
        I2S_PIN_NO_CHANGE;

    pin_config.bck_io_num =
        I2S_BCLK;

    pin_config.ws_io_num =
        I2S_WS;

    pin_config.data_out_num =
        I2S_PIN_NO_CHANGE;

    pin_config.data_in_num =
        I2S_DOUT;


    result = i2s_set_pin(
        I2S_PORT,
        &pin_config
    );

    if (result != ESP_OK)
    {
        Serial.print(
            "I2S pin configuration FAILED: "
        );
        Serial.println(result);

        while (true)
        {
            digitalWrite(RED_LED, LOW);
            delay(500);

            digitalWrite(RED_LED, HIGH);
            delay(500);
        }
    }


    // Clear old DMA data
    i2s_zero_dma_buffer(I2S_PORT);

    Serial.println(
        "I2S initialized successfully."
    );

    Serial.println();
}


// ============================================================
//                  CAPTURE MICROPHONE AUDIO
// ============================================================

bool captureAudio()
{
    if (audioBuffer == nullptr)
    {
        return false;
    }

    size_t captured = 0;

    while (captured < AUDIO_SAMPLES)
    {
        int32_t rawSample = 0;
        size_t bytesRead = 0;

        esp_err_t result = i2s_read(
            I2S_PORT,
            &rawSample,
            sizeof(rawSample),
            &bytesRead,
            portMAX_DELAY
        );

        if (result != ESP_OK)
        {
            Serial.println(
                "I2S read error!"
            );

            return false;
        }

        if (bytesRead != sizeof(rawSample))
        {
            continue;
        }


        // ----------------------------------------------------
        // INMP441:
        // 24-bit audio inside 32-bit I2S word
        //
        // >>16 gives the required signed 16-bit range.
        // ----------------------------------------------------

        int32_t converted =
            rawSample >> 16;

        if (converted > 32767)
        {
            converted = 32767;
        }

        if (converted < -32768)
        {
            converted = -32768;
        }

        audioBuffer[captured] =
            (int16_t)converted;

        captured++;
    }

    return true;
}


// ============================================================
//                    AUDIO DIAGNOSTICS
// ============================================================

void printAudioDiagnostics()
{
    int16_t minimum = 32767;
    int16_t maximum = -32768;

    double sumAbs = 0.0;
    double sumSquare = 0.0;

    uint32_t clippedSamples = 0;


    for (size_t i = 0;
         i < AUDIO_SAMPLES;
         i++)
    {
        int32_t sample =
            audioBuffer[i];


        if (sample < minimum)
        {
            minimum = sample;
        }

        if (sample > maximum)
        {
            maximum = sample;
        }


        sumAbs +=
            abs(sample);


        sumSquare +=
            ((double)sample *
             (double)sample);


        if (
            sample >= 32767 ||
            sample <= -32768
        )
        {
            clippedSamples++;
        }
    }


    double averageAbs =
        sumAbs / AUDIO_SAMPLES;


    double rms =
        sqrt(
            sumSquare /
            AUDIO_SAMPLES
        );


    Serial.println();
    Serial.println(
        "----- AUDIO DIAGNOSTICS -----"
    );


    Serial.print("Minimum: ");
    Serial.println(minimum);


    Serial.print("Maximum: ");
    Serial.println(maximum);


    Serial.print(
        "Average absolute: "
    );

    Serial.println(
        averageAbs,
        2
    );


    Serial.print("RMS: ");
    Serial.println(
        rms,
        2
    );


    Serial.print(
        "Clipped samples: "
    );

    Serial.print(
        clippedSamples
    );

    Serial.print(" / ");

    Serial.println(
        AUDIO_SAMPLES
    );


    Serial.println(
        "------------------------------"
    );
}


// ============================================================
//                       SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    delay(1500);


    Serial.println();
    Serial.println(
        "********************************"
    );

    Serial.println(
        "       PILOT KWS - ESP32"
    );

    Serial.println(
        "********************************"
    );

    Serial.println();


    // --------------------------------------------------------
    // LEDs
    // --------------------------------------------------------

    pinMode(
        RED_LED,
        OUTPUT
    );

    pinMode(
        BLUE_LED,
        OUTPUT
    );


    digitalWrite(
        RED_LED,
        LOW
    );

    digitalWrite(
        BLUE_LED,
        LOW
    );


    // --------------------------------------------------------
    // Allocate audio buffer
    // --------------------------------------------------------

    audioBuffer =
        (int16_t *)malloc(
            AUDIO_SAMPLES *
            sizeof(int16_t)
        );


    if (audioBuffer == nullptr)
    {
        Serial.println();
        Serial.println(
            "ERROR: Audio buffer allocation FAILED!"
        );


        while (true)
        {
            digitalWrite(
                RED_LED,
                HIGH
            );

            delay(200);

            digitalWrite(
                RED_LED,
                LOW
            );

            delay(200);
        }
    }


    Serial.print(
        "Audio buffer allocated: "
    );

    Serial.print(
        AUDIO_SAMPLES *
        sizeof(int16_t)
    );

    Serial.println(
        " bytes"
    );


    // --------------------------------------------------------
    // Start microphone
    // --------------------------------------------------------

    setupI2S();


    // --------------------------------------------------------
    // Start PILOT KWS model
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "Initializing PILOT KWS model..."
    );


    if (!PILOT_KWS.begin())
    {
        Serial.println();
        Serial.println(
            "ERROR: PILOT KWS initialization FAILED!"
        );


        while (true)
        {
            digitalWrite(
                RED_LED,
                HIGH
            );

            delay(100);

            digitalWrite(
                RED_LED,
                LOW
            );

            delay(100);
        }
    }


    Serial.println(
        "PILOT KWS initialized successfully."
    );


    Serial.println();
    Serial.println(
        "System ready."
    );

    Serial.println(
        "Listening for: PILOT"
    );

    Serial.println();


    // RED = listening
    digitalWrite(
        RED_LED,
        HIGH
    );
}


// ============================================================
//                        MAIN LOOP
// ============================================================

void loop()
{
    // --------------------------------------------------------
    // Blue LED timeout
    // --------------------------------------------------------

    if (
        blueLedUntil != 0 &&
        millis() >= blueLedUntil
    )
    {
        digitalWrite(
            BLUE_LED,
            LOW
        );

        blueLedUntil = 0;


        Serial.println();
        Serial.println(
            "BLUE LED OFF"
        );
        Serial.println();
    }


    // --------------------------------------------------------
    // Capture audio
    // --------------------------------------------------------

    Serial.println(
        "Listening..."
    );


    bool captureOK =
        captureAudio();


    if (!captureOK)
    {
        Serial.println(
            "Audio capture FAILED!"
        );

        delay(100);

        return;
    }


    Serial.println(
        "Audio captured."
    );


    // --------------------------------------------------------
    // First samples
    // --------------------------------------------------------

    Serial.print(
        "First samples: "
    );


    for (int i = 0;
         i < 10;
         i++)
    {
        Serial.print(
            audioBuffer[i]
        );


        if (i < 9)
        {
            Serial.print(", ");
        }
    }


    Serial.println();


    // --------------------------------------------------------
    // Audio diagnostics
    // --------------------------------------------------------

    printAudioDiagnostics();


    // --------------------------------------------------------
    // Run PILOT KWS
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "Running PILOT detection..."
    );


    bool processOK =
        PILOT_KWS.processAudio(
            audioBuffer,
            AUDIO_SAMPLES
        );


    if (!processOK)
    {
        Serial.println();
        Serial.println(
            "ERROR: PILOT KWS processing FAILED!"
        );
        Serial.println();

        return;
    }


    // --------------------------------------------------------
    // Get confidence values
    // --------------------------------------------------------

    float backgroundConfidence =
        PILOT_KWS.getBackgroundConfidence();


    float negativeConfidence =
        PILOT_KWS.getNegativeConfidence();


    float pilotConfidence =
        PILOT_KWS.getPilotConfidence();


    // --------------------------------------------------------
    // Print result
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "========== KWS RESULT =========="
    );


    Serial.print(
        "Background: "
    );

    Serial.println(
        backgroundConfidence,
        6
    );


    Serial.print(
        "Negative:   "
    );

    Serial.println(
        negativeConfidence,
        6
    );


    Serial.print(
        "PILOT:      "
    );

    Serial.println(
        pilotConfidence,
        6
    );


    Serial.println(
        "================================"
    );


    // --------------------------------------------------------
    // PILOT detection
    //
    // The library threshold is:
    //
    //     PILOT >= 0.03
    //
    // Therefore BLUE LED turns ON whenever
    // isPilotDetected() returns true.
    // --------------------------------------------------------

    if (
        PILOT_KWS.isPilotDetected()
    )
    {
        Serial.println();
        Serial.println(
            "********************************"
        );

        Serial.println(
            "        PILOT DETECTED!"
        );

        Serial.println(
            "********************************"
        );

        Serial.println();


        // BLUE LED ON
        digitalWrite(
            BLUE_LED,
            HIGH
        );


        // Keep ON for 5 seconds
        blueLedUntil =
            millis() + 5000;


        Serial.println(
            "BLUE LED ON FOR 5 SECONDS"
        );

        Serial.println();
    }
    else
    {
        Serial.println();
        Serial.println(
            "No PILOT detected."
        );
        Serial.println();
    }
}
