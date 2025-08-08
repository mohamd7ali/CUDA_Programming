
#include "finite_field.h"
using namespace std;
#include <iostream>
FiniteField::FiniteField(/* args */)
{
    int32_t degree = 8;
    int32_t p_decimal = 357;
    generateFieldParams(alpha_to, index_of, degree, p_decimal);
}

FiniteField::~FiniteField()
{
    if (alpha_to != NULL){					delete[] alpha_to;									    alpha_to = NULL;}
    if (index_of != NULL){					delete[] index_of;									    index_of = NULL;}
}


void FiniteField::generateFieldParams(uint8_t* &alpha_to, uint8_t* &index_of, int32_t degree, int32_t p_decimal)
{
	int32_t n = (1 << degree);

	alpha_to = new uint8_t[n];
	index_of = new uint8_t[n];
	
	int32_t *p = new int32_t[degree + 1];

	for (int32_t i = 0; i < degree + 1; i++)
	{
		p[i] = (p_decimal >> i) & 1;
	}
	//generate GF(2^m) from the irreducible polynomial p(X)
	int32_t mask = 1;
	alpha_to[degree] = 0;
	for (int32_t i = 0; i < degree; i++)
	{
		alpha_to[i] = mask;
		index_of[alpha_to[i]] = i;
		if (p[i] != 0)
			alpha_to[degree] ^= mask;
		mask <<= 1;
	}
	delete[] p;

	index_of[alpha_to[degree]] = degree;
	mask >>= 1;
	for (int32_t i = degree + 1; i < n-1; i++)
	{
		if (alpha_to[i - 1] >= mask)
			alpha_to[i] = uint32_t(alpha_to[degree]) ^ ((mask ^ alpha_to[i - 1]) << 1);
		else
			alpha_to[i] = alpha_to[i - 1] << 1;
		index_of[alpha_to[i]] = i;
	}
	index_of[0] = uint8_t(-1);
}


uint8_t FiniteField::multiply(uint8_t a, uint8_t b, uint8_t* alpha_to, uint8_t* index_of)
{
	if (a == 0 || b == 0)
	{
		return 0;
	}
	else
	{
		return alpha_to[(uint32_t(index_of[a]) + uint32_t(index_of[b])) % 255];
	}
}

uint8_t FiniteField::sum(uint8_t a, uint8_t b)
{
	return a^b;
}
