#ifndef __DISNVM_EXAMPLES_UTIL_H__
#define __DISNVM_EXAMPLES_UTIL_H__

#include <nvm_types.h>
#include <stdint.h>
#include <getopt.h>
#include <stdio.h>


/*
 * Generate a random identifier.
 */
uint16_t random_id();


/*
 * Parse an uint64_t from a string.
 */
int parse_u64(const char* str, uint64_t* number, int base);


/*
 * Parse an uint32_t from a string.
 */
int parse_u32(const char* str, uint32_t* number, int base);


/*
 * Parse an uint16_t from a string.
 */
int parse_u16(const char* str, uint16_t* number, int base);


/*
 * Pretty print controller information.
 */
void print_ctrl_info(FILE* fp, const struct nvm_ctrl_info* info);


/*
 * Pretty print namespace information.
 */
void print_ns_info(FILE* fp, const struct nvm_ns_info* info);

#endif // __DISNVM_EXAMPLES_UTIL_H__
