NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -lineinfo -Xcompiler -Wall,-Wextra -Ithird_party/cjson
CC = gcc
CFLAGS = -O2 -Wall -Wextra
BUILD = build
MODEL = models/Qwen3-0.6B

SRCS = $(wildcard src/*.cu)
OBJS = $(patsubst src/%.cu,$(BUILD)/%.o,$(SRCS)) $(BUILD)/cJSON.o

$(BUILD)/engine: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD)/%.o: src/%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $<

$(BUILD)/cJSON.o: third_party/cjson/cJSON.c third_party/cjson/cJSON.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

ref:
	uv run ref/dump.py --model $(MODEL)

check-weights: $(BUILD)/engine
	./$(BUILD)/engine $(MODEL) --checksums | sort > $(BUILD)/checksums_engine.txt
	uv run ref/weight_checksums.py $(MODEL) | sort > $(BUILD)/checksums_ref.txt
	diff $(BUILD)/checksums_engine.txt $(BUILD)/checksums_ref.txt
	@echo "all $$(wc -l < $(BUILD)/checksums_ref.txt) tensors match"

clean:
	rm -rf $(BUILD)

.PHONY: clean ref check-weights

-include $(OBJS:.o=.d)
