#include "helper.h"

class FiniteField
{
private:
    void generateFieldParams(uint8_t* &alpha_to, uint8_t* &index_of, int32_t degree, int32_t p_decimal);
public:
    FiniteField(/* args */);
    ~FiniteField();

    uint8_t multiply(uint8_t a, uint8_t b, uint8_t* alpha_to, uint8_t* index_of);
    uint8_t sum(uint8_t a, uint8_t b);

    uint8_t* alpha_to;
    uint8_t* index_of;
};
