CC = gcc
CFLAGS = -Wall -Wextra -g -O3 -mcpu=cortex-a76 -Isrc
CORE_SRCS = src/matrix.c src/layers.c src/model.c src/neon_kernel.c
CORE_OBJS = $(CORE_SRCS:.c=.o)

.PHONY: all clean

all: neonnet neonnet_bench

neonnet: $(CORE_OBJS) src/main.o
	$(CC) $(CORE_OBJS) src/main.o -o neonnet -lm

neonnet_bench: $(CORE_OBJS) bench/benchmark.o
	$(CC) $(CORE_OBJS) bench/benchmark.o -o neonnet_bench -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(CORE_OBJS) src/main.o bench/benchmark.o neonnet neonnet_bench
