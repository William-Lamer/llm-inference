NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -lineinfo -Xcompiler -Wall,-Wextra
BUILD = build

SRCS = $(wildcard src/*.cu)
OBJS = $(patsubst src/%.cu,$(BUILD)/%.o,$(SRCS))

$(BUILD)/engine: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD)/%.o: src/%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: clean

-include $(OBJS:.o=.d)
