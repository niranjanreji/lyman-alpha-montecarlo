#define  PHYSICS                 HD
#define  DIMENSIONS              1
#define  COMPONENTS              1
#define  GEOMETRY                SPHERICAL
#define  BODY_FORCE              NO
#define  COOLING                 NO
#define  INTERPOLATION           LINEAR
#define  TIME_STEPPING           RK2
#define  DIMENSIONAL_SPLITTING   YES
#define  NTRACER                 0
#define  USER_DEF_PARAMETERS     3

/* -- physics dependent declarations -- */

#define    EOS                     IDEAL
#define    ENTROPY_SWITCH          NO
#define    THERMAL_CONDUCTION      NO
#define    VISCOSITY               NO
#define    ROTATING_FRAME          NO

/* -- pointers to user-def parameters -- */

#define  RHO_WIND           0
#define  V_WIND             1
#define  CS_WIND            2

/* -- supplementary constants (user editable) -- */ 

#define  INITIAL_SMOOTHING     NO
#define  WARNING_MESSAGES      YES
#define  PRINT_TO_FILE         YES
#define  INTERNAL_BOUNDARY     YES
#define  SHOCK_FLATTENING      MULTID
#define  ARTIFICIAL_VISCOSITY  NO
#define  CHAR_LIMITING         YES
#define  LIMITER               VANALBADA_LIM
