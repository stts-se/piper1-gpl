%module espeak_ng

%typemap(out) TupleResult {
    $result = Py_BuildValue("(s,s,i)", $1.phonemes, $1.remaining_text, $1.terminator);
}

%{
extern int espeak_Initialize(int output, int buflength, const char *path, int options);
extern const char *espeak_TextToPhonemesWithTerminator(const void **textptr, int textmode, int phonememode, int *terminator);
extern int espeak_SetVoiceByName(const char *name);
extern int espeak_Terminate(void);

typedef struct {
    const char* phonemes;
    const char* remaining_text;
    int terminator;
} TupleResult;

// Wrap away the const void** hell
TupleResult py_espeak_TextToPhonemesWithTerminator(const char *text, int textmode, int phonememode) {
    const void *ptr = text;
    int terminator = 0;

    const char *phonemes = espeak_TextToPhonemesWithTerminator(&ptr, textmode, phonememode, &terminator);
    TupleResult result = { phonemes, (const char *)ptr, terminator };
    return result;
}
%}

/* These declarations are used to generate the Python bindings */
extern int espeak_Initialize(int output, int buflength, const char *path, int options);
/* extern const char *espeak_TextToPhonemesWithTerminator(const void **textptr, int textmode, int phonememode, int *terminator); */
extern int espeak_SetVoiceByName(const char *name);
extern int espeak_Terminate(void);

TupleResult py_espeak_TextToPhonemesWithTerminator(const char *text, int textmode, int phonememode);
