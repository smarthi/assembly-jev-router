.PHONY: all test clean

all: test_jev_router

test_jev_router: jev_router.s test_jev_router.c
	clang -Wall -Wextra -Werror jev_router.s test_jev_router.c -o $@

test: test_jev_router
	./test_jev_router

clean:
	rm -f test_jev_router
