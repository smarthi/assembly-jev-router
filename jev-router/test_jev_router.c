#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    uint64_t class_count;
    uint64_t feature_count;
    const int32_t *weights;
    const int64_t *biases;
    const void *const *handlers;
} jev_model;

typedef struct {
    uint64_t class_id;
    int64_t best_score;
    int64_t runner_up_score;
    uint64_t confidence_bps;
} jev_result;

extern int64_t jev_classify(const jev_model *, const int32_t *, jev_result *);
extern int64_t jev_route(const jev_model *, const int32_t *, jev_result *, void *);

enum { ROUTE_READ, ROUTE_WRITE, ROUTE_REVIEW };

static int64_t on_read(void *ctx, const jev_result *result) {
    (void)result;
    return *(int64_t *)ctx + 100;
}

static int64_t on_write(void *ctx, const jev_result *result) {
    (void)result;
    return *(int64_t *)ctx + 200;
}

static int64_t on_review(void *ctx, const jev_result *result) {
    (void)result;
    return *(int64_t *)ctx + 300;
}

int main(void) {
    // Features: [read signal, write signal, ambiguity signal].
    static const int32_t weights[] = {
         8, -4, -1,                 // ROUTE_READ
        -4,  8, -1,                 // ROUTE_WRITE
        -1, -1,  7,                 // ROUTE_REVIEW
    };
    static const int64_t biases[] = { 0, 0, 1 };
    static const void *const handlers[] = { on_read, on_write, on_review };
    const jev_model model = { 3, 3, weights, biases, handlers };
    jev_result result;

    const int32_t read_case[] = { 5, 0, 0 };
    assert(jev_classify(&model, read_case, &result) == 0);
    assert(result.class_id == ROUTE_READ);
    assert(result.best_score == 40);
    assert(result.confidence_bps >= 5000 && result.confidence_bps <= 10000);

    const int32_t write_case[] = { 0, 4, 0 };
    int64_t context = 7;
    assert(jev_route(&model, write_case, &result, &context) == 207);
    assert(result.class_id == ROUTE_WRITE);

    const int32_t unclear_case[] = { 1, 1, 5 };
    assert(jev_route(&model, unclear_case, &result, &context) == 307);
    assert(result.class_id == ROUTE_REVIEW);

    jev_model no_handlers = model;
    no_handlers.handlers = NULL;
    assert(jev_route(&no_handlers, read_case, &result, &context) == -2);
    assert(jev_classify(NULL, read_case, &result) == -1);

    printf("class=%" PRIu64 " score=%" PRId64 " confidence=%" PRIu64 ".%02" PRIu64 "%%\n",
           result.class_id, result.best_score,
           result.confidence_bps / 100, result.confidence_bps % 100);
    puts("Jev-style AArch64 classifier/router tests passed");
    return 0;
}
