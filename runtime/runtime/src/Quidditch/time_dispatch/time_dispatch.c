
#include "time_dispatch.h"

#include <team_decls.h>
unsigned long myrtle_actual_cycles[5][2] = {
    {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};

void myrtle_record_cycles(uint32_t kernel, uint32_t atEnd) {
  uint32_t register r;
  asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
  myrtle_actual_cycles[kernel][atEnd] = r;
}
