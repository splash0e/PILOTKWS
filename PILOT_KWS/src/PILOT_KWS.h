#ifndef PILOT_KWS_H
#define PILOT_KWS_H

#include <Arduino.h>

class PILOT_KWS_Class {
public:

    bool begin();

    bool processAudio(const int16_t* audio, size_t samples);

    bool isPilotDetected();

    float getPilotConfidence();

    float getBackgroundConfidence();

    float getNegativeConfidence();

private:

    float pilotConfidence = 0.0f;
    float backgroundConfidence = 0.0f;
    float negativeConfidence = 0.0f;
};

extern PILOT_KWS_Class PILOT_KWS;

#endif
