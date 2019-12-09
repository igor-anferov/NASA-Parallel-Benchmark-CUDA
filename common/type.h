#ifndef __TYPE_H__
#define __TYPE_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __cplusplus
typedef enum { false, true } logical;
#else
typedef bool logical;
#endif

typedef struct { 
  double real;
  double imag;
} dcomplex;


#define min(x,y)    ((x) < (y) ? (x) : (y))
#define max(x,y)    ((x) > (y) ? (x) : (y))

#ifdef __cplusplus
}
#endif

#endif //__TYPE_H__
