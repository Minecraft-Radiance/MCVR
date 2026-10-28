// FSR1 filters a bounded perceptual signal. Preserve scene-linear HDR around
// EASU/RCAS with a reversible mapping because this module precedes tone mapping.
vec4 fsr1Encode(vec4 radiance) {
    radiance=mix(clamp(radiance,vec4(0),vec4(65504)),vec4(0),isnan(radiance));
    return log2(vec4(1)+radiance)/16.0;
}
vec3 fsr1Decode(vec3 signal) {
    signal=clamp(signal,vec3(0),vec3(1));
    return min(exp2(signal*16.0)-vec3(1),vec3(65504));
}
