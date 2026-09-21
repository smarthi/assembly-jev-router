// Jev-style typed classification and routing for Apple Silicon.
// AArch64 / macOS, AAPCS64. No runtime or allocation dependencies.
//
// This is a local linear decision engine, not TypeSafe AI's proprietary Jev
// model. It maps an int32 feature vector to a predefined class, reports the
// top two scores and a confidence heuristic, and can invoke a typed route.
//
// struct jev_model {                 // offsets
//     uint64_t class_count;          //  0
//     uint64_t feature_count;        //  8
//     const int32_t *weights;        // 16, class-major matrix
//     const int64_t *biases;         // 24, one per class
//     const void *const *handlers;   // 32, one function pointer per class
// };
//
// struct jev_result {
//     uint64_t class_id;             //  0
//     int64_t best_score;            //  8
//     int64_t runner_up_score;       // 16
//     uint64_t confidence_bps;       // 24, 5000..10000
// };
//
// int64_t jev_classify(model=x0, features=x1, result=x2)
//   Returns 0 on success or -1 for invalid input.
//
// int64_t jev_route(model=x0, features=x1, result=x2, context=x3)
//   Classifies, then calls handlers[result.class_id](context, result).
//   Returns the handler's value, -1 for invalid input, or -2 for no handler.

.text
.p2align 2

.globl _jev_classify
.globl _jev_route

.equ MODEL_CLASSES,   0
.equ MODEL_FEATURES,  8
.equ MODEL_WEIGHTS,   16
.equ MODEL_BIASES,    24
.equ MODEL_HANDLERS,  32

.equ RESULT_CLASS,    0
.equ RESULT_BEST,     8
.equ RESULT_SECOND,   16
.equ RESULT_CONF,     24

_jev_classify:
    // Validate pointers and non-empty dimensions.
    cbz     x0, .Lclassify_invalid
    cbz     x1, .Lclassify_invalid
    cbz     x2, .Lclassify_invalid
    ldr     x3, [x0, #MODEL_CLASSES]
    ldr     x4, [x0, #MODEL_FEATURES]
    cbz     x3, .Lclassify_invalid
    cbz     x4, .Lclassify_invalid
    ldr     x5, [x0, #MODEL_WEIGHTS]
    ldr     x6, [x0, #MODEL_BIASES]
    cbz     x5, .Lclassify_invalid
    cbz     x6, .Lclassify_invalid

    // best and runner-up start at INT64_MIN.
    mov     x7, #1
    lsl     x7, x7, #63
    mov     x8, x7                  // best score
    mov     x9, x7                  // second-best score
    mov     x10, xzr                // best class index
    mov     x11, xzr                // current class index

.Lclass_loop:
    ldr     x12, [x6, x11, lsl #3] // accumulator = class bias
    mov     x13, x4                 // remaining features
    mov     x14, x1                 // feature cursor

.Ldot_loop:
    ldrsw   x15, [x5], #4           // signed int32 weight
    ldrsw   x16, [x14], #4          // signed int32 feature
    madd    x12, x15, x16, x12
    subs    x13, x13, #1
    b.ne    .Ldot_loop

    cmp     x12, x8
    b.le    .Lmaybe_second
    mov     x9, x8
    mov     x8, x12
    mov     x10, x11
    b       .Lnext_class

.Lmaybe_second:
    cmp     x12, x9
    csel    x9, x12, x9, gt

.Lnext_class:
    add     x11, x11, #1
    cmp     x11, x3
    b.lo    .Lclass_loop

    str     x10, [x2, #RESULT_CLASS]
    str     x8, [x2, #RESULT_BEST]

    // With one class there is no runner-up and confidence is total.
    cmp     x3, #1
    b.ne    .Lconfidence
    str     x8, [x2, #RESULT_SECOND]
    mov     x17, #10000
    str     x17, [x2, #RESULT_CONF]
    mov     x0, xzr
    ret

.Lconfidence:
    str     x9, [x2, #RESULT_SECOND]

    // Confidence heuristic:
    //   5000 + 5000 * abs(best - second)
    //                 / (abs(best) + abs(second) + 1)
    // Floating-point is used only for this normalization; classification is
    // entirely signed integer arithmetic and therefore deterministic.
    scvtf   d0, x8
    scvtf   d1, x9
    fsub    d2, d0, d1
    fabs    d2, d2
    fabs    d0, d0
    fabs    d1, d1
    fadd    d0, d0, d1
    fmov    d1, #1.0
    fadd    d0, d0, d1
    fdiv    d2, d2, d0
    mov     x17, #5000
    ucvtf   d1, x17
    fmadd   d2, d2, d1, d1
    fcvtzu  x17, d2
    mov     x18, #10000
    cmp     x17, x18
    csel    x17, x17, x18, lo
    str     x17, [x2, #RESULT_CONF]
    mov     x0, xzr
    ret

.Lclassify_invalid:
    mov     x0, #-1
    ret

_jev_route:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]

    mov     x19, x0                 // model survives classification
    mov     x20, x2                 // result survives classification
    mov     x21, x3                 // handler context
    bl      _jev_classify
    cbnz    x0, .Lroute_return       // propagate -1

    ldr     x4, [x19, #MODEL_HANDLERS]
    cbz     x4, .Lroute_missing
    ldr     x5, [x20, #RESULT_CLASS]
    ldr     x5, [x4, x5, lsl #3]
    cbz     x5, .Lroute_missing

    mov     x0, x21                 // handler(context, result)
    mov     x1, x20
    blr     x5
    b       .Lroute_return

.Lroute_missing:
    mov     x0, #-2
.Lroute_return:
    ldp     x19, x20, [sp, #16]
    ldr     x21, [sp, #32]
    ldp     x29, x30, [sp], #48
    ret
