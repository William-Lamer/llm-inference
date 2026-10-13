NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -lineinfo -Xcompiler -Wall,-Wextra -Ithird_party/cjson
CC = gcc
CFLAGS = -O2 -Wall -Wextra
BUILD = build
MODEL = models/Qwen3-0.6B
REF = dumps/$(notdir $(MODEL))
OUT = $(BUILD)/dumps/$(notdir $(MODEL))
GREEDY = $(BUILD)/greedy/$(notdir $(MODEL))
LLAMACPP = $(HOME)/projects/llama.cpp

BENCH_FLAGS = $(if $(wildcard $(LLAMACPP)/build/bin/llama-bench),--llamacpp $(LLAMACPP))

SRCS = $(wildcard src/*.cu)
OBJS = $(patsubst src/%.cu,$(BUILD)/%.o,$(SRCS)) $(BUILD)/cJSON.o

$(BUILD)/engine: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD)/%.o: src/%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $<

$(BUILD)/bandwidth: bench/bandwidth.cu src/common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -Isrc -o $@ $<

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

# Greedy runs each prompt twice through the 2 modes.
check: $(BUILD)/engine
	rm -rf $(OUT) $(GREEDY)
	mkdir -p $(OUT) $(GREEDY)
	for p in $(REF)/p*; do \
	    d=$(OUT)/$$(basename $$p); \
	    ./$(BUILD)/engine $(MODEL) --tokens $$p/tokens.txt --dump $$d > /dev/null 2> $$d.log || { cat $$d.log; exit 1; }; \
	done
	uv run ref/compare.py $(REF) $(OUT)
	for p in $(REF)/p*; do \
	    d=$(GREEDY)/$$(basename $$p); \
	    mkdir -p $$d; \
	    ./$(BUILD)/engine $(MODEL) --tokens $$p/tokens.txt --max-new $$(wc -w < $$p/greedy.txt) > $$d/generated.txt 2> $$d/engine.log || { cat $$d/engine.log; exit 1; }; \
	    cat $$p/tokens.txt $$p/greedy.txt > $$d/forced_tokens.txt; \
	    ./$(BUILD)/engine $(MODEL) --tokens $$d/forced_tokens.txt --dump $$d > /dev/null 2>> $$d/engine.log || { cat $$d/engine.log; exit 1; }; \
	done
	uv run ref/greedy.py $(REF) $(GREEDY)

calibrate:
	uv run ref/calibrate.py --model $(MODEL)
	uv run ref/compare.py -v $(REF) $(BUILD)/calib/$(notdir $(MODEL))

bench: $(BUILD)/engine $(BUILD)/bandwidth
	uv run bench/bench.py --model $(MODEL) $(BENCH_FLAGS)

# Same as bench and profile, then appends the numbers to bench/results.csv
record: $(BUILD)/engine $(BUILD)/bandwidth
	uv run bench/bench.py --model $(MODEL) $(BENCH_FLAGS) --record
	uv run bench/breakdown.py --model $(MODEL) --record

# Same as record, with the displays off and the GPU checked idle first
record-quiet: $(BUILD)/engine $(BUILD)/bandwidth
	bench/quiet.sh $(MAKE) record

profile: $(BUILD)/engine
	uv run bench/breakdown.py --model $(MODEL)

.PHONY: clean ref check-weights check calibrate bench record record-quiet profile

-include $(OBJS:.o=.d)
