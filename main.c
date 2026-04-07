#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "cubiomes/biomenoise.h"
#include "cubiomes/finders.h"
#include "cubiomes/generator.h"

#define THREADS 12
#define BATCH_SIZE 16384

typedef struct {
  BiomeNoise *t, *h, *c, *e, *w;
} ClimateNoises;

typedef struct {
  double t, h, c, e, w;
} ClimateValues;

static atomic_uint_fast64_t next_seed = 0;

static void print_result(uint64_t seed, int x, int z, double value) {
  printf("%" PRId64 " %d %d %f\n", (int64_t)seed, x * 4, z * 4, value);
}

static int check_offsets(uint64_t seed) {
  Xoroshiro x1, x2, x3;
  xSetSeed(&x1, seed);
  uint64_t xlo = xNextLong(&x1);
  uint64_t xhi = xNextLong(&x1);
  x2.lo = xlo ^ 0xefc8ef4d36102b34;
  x2.hi = xhi ^ 0x1beeeb324a0f24ea;
  xlo = xNextLong(&x2);
  xhi = xNextLong(&x2);
  x3.lo = xlo ^ 0xf11268128982754f;
  x3.hi = xhi ^ 0x257a1d670430b0aa;
  xSkipN(&x3, 1);
  float off0a =
      fabsf((xNextLong(&x3) >> 32 & 0xFFFFFF) * (1.0f / 16777216.0f) - 0.5f);
  x3.lo = xlo ^ 0xe51c98ce7d1de664;
  x3.hi = xhi ^ 0x5f9478a733040c45;
  xSkipN(&x3, 1);
  float off1a =
      fabsf((xNextLong(&x3) >> 32 & 0xFFFFFF) * (1.0f / 16777216.0f) - 0.5f);
  xlo = xNextLong(&x2);
  xhi = xNextLong(&x2);
  x3.lo = xlo ^ 0xf11268128982754f;
  x3.hi = xhi ^ 0x257a1d670430b0aa;
  xSkipN(&x3, 1);
  float off0b =
      fabsf((xNextLong(&x3) >> 32 & 0xFFFFFF) * (1.0f / 16777216.0f) - 0.5f);
  x3.lo = xlo ^ 0xe51c98ce7d1de664;
  x3.hi = xhi ^ 0x5f9478a733040c45;
  xSkipN(&x3, 1);
  float off1b =
      fabsf((xNextLong(&x3) >> 32 & 0xFFFFFF) * (1.0f / 16777216.0f) - 0.5f);
  return off0a + off0b + off1a + off1b < 0.2f;
}

static double sample(BiomeNoise *bn, int a, int b, int x, int z) {
  double v = 0;
  if (a) {
    OctaveNoise *on = &bn->climate[bn->nptype].octA;
    for (int i = 0; i < on->octcnt; i++) {
      if (a & (1 << i)) {
        PerlinNoise *pn = on->octaves + i;
        double l = pn->lacunarity;
        v += samplePerlin(pn, x * l, 0, z * l, 0, 0) * pn->amplitude;
      }
    }
  }
  if (b) {
    OctaveNoise *on = &bn->climate[bn->nptype].octB;
    for (int i = 0; i < on->octcnt; i++) {
      if (b & (1 << i)) {
        PerlinNoise *pn = on->octaves + i;
        double l = pn->lacunarity * (337.0 / 331.0);
        v += samplePerlin(pn, x * l, 0, z * l, 0, 0) * pn->amplitude;
      }
    }
  }
  return v * bn->climate[bn->nptype].amplitude;
}

static void lattice(ClimateNoises *n, uint64_t seed, double max_a, int sign,
                    int x, int z) {
  setClimateParaSeed(n->w, seed, 0, NP_WEIRDNESS, 6);
  for (int x1 = x - 7340032; x1 <= x + 7340032; x1 += 32768)
    for (int z1 = z - 7340032; z1 <= z + 7340032; z1 += 32768) {
      if (sample(n->w, 0b000, 0b011, x1, z1) * sign + max_a < 2.2) {
        continue;
      }
      double max = 0.0;
      int max_x = 0;
      int max_z = 0;
      for (int x2 = x1 - 32; x2 <= x1 + 32; x2 += 1)
        for (int z2 = z1 - 32; z2 <= z1 + 32; z2 += 1) {
          double value = sample(n->w, 0b111, 0b111, x2, z2) * sign;
          if (value > max) {
            max = value;
            max_x = x2;
            max_z = z2;
          }
        }
      if (max > 2.4) {
        print_result(seed, max_x * 4, max_z * 4, max * sign);
      }
    }
}

static void check(ClimateNoises *n, uint64_t seed) {
  if (!check_offsets(seed)) {
    return;
  }
  setClimateParaSeed(n->w, seed, 0, NP_WEIRDNESS, 1);
  for (int x1 = -1024; x1 <= 1024; x1 += 256)
    for (int z1 = -1024; z1 <= 1024; z1 += 256) {
      double value = sample(n->w, 0b1, 0b0, x1, z1);
      int sign = (value > 0.4) - (value < -0.4);
      if (sign == 0) {
        continue;
      }
      for (int x2 = x1 - 64; x2 <= x1 + 64; x2 += 16)
        for (int z2 = z1 - 64; z2 <= z1 + 64; z2 += 16) {
          if (sample(n->w, 0b1, 0b0, x2, z2) * sign < 0.6) {
            continue;
          }
          setClimateParaSeed(n->w, seed, 0, NP_WEIRDNESS, 5);
          double max = 0.0;
          int max_x = 0;
          int max_z = 0;
          for (int x3 = x2 - 32; x3 <= x2 + 32; x3 += 4)
            for (int z3 = z2 - 32; z3 <= z2 + 32; z3 += 4) {
              double value = sample(n->w, 0b111, 0b000, x3, z3) * sign;
              if (value > max) {
                max = value;
                max_x = x3;
                max_z = z3;
              }
            }
          if (max > 1.2) {
            lattice(n, seed, max, sign, max_x, max_z);
          }
          return;
        }
    }
}

static void *worker(void *arg) {
  Generator g;
  setupGenerator(&g, MC_NEWEST, 0);
  BiomeNoise t, h, c, e, w;
  t = h = c = e = w = g.bn;
  ClimateNoises n = {&t, &h, &c, &e, &w};
  while (1) {
    uint64_t start = atomic_fetch_add(&next_seed, BATCH_SIZE);
    uint64_t end = start + BATCH_SIZE;
    for (uint64_t seed = start; seed < end; seed++) {
      check(&n, seed);
    }
  }
  return NULL;
}

int main(void) {
  srand((unsigned int)time(NULL));
  uint64_t seed = 0;
  for (int i = 0; i < 4; i++) {
    seed = (seed << 16) | (uint64_t)(rand() & 0xFFFF);
  }
  atomic_store(&next_seed, seed);
  pthread_t threads[THREADS];
  for (int i = 0; i < THREADS; i++) {
    pthread_create(&threads[i], NULL, worker, NULL);
  }
  for (int i = 0; i < THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  return 0;
}
