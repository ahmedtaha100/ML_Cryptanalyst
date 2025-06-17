"""
Constant-time implementation examples from the paper.
These show how to transform vulnerable code into constant-time versions.
"""

import numpy as np
from typing import Union, Tuple, List, Dict


class ConstantTimeImplementations:
    """Examples of constant-time transformations for cryptographic operations."""

    @staticmethod
    def vulnerable_conditional_copy(secret_bit: int, value_if_true: int,
                                   value_if_false: int) -> int:
        """VULNERABLE: Execution time depends on secret bit value."""
        if secret_bit:
            result = value_if_true
        else:
            result = value_if_false
        return result

    @staticmethod
    def constant_time_conditional_copy(secret_bit: int, value_if_true: int,
                                      value_if_false: int) -> int:
        """SECURE: Constant-time conditional copy using arithmetic."""
        secret_bit = secret_bit & 1
        result = (secret_bit * value_if_true) | ((1 - secret_bit) * value_if_false)
        return result

    @staticmethod
    def constant_time_compare(a: bytes, b: bytes) -> bool:
        """Constant-time byte string comparison."""
        if len(a) != len(b):
            return False
        result = 0
        for byte_a, byte_b in zip(a, b):
            result |= byte_a ^ byte_b
        return result == 0

    @staticmethod
    def vulnerable_array_lookup(secret_index: int, array: List[int]) -> int:
        """VULNERABLE: Direct array access can leak index through cache timing."""
        return array[secret_index]

    @staticmethod
    def constant_time_array_lookup(secret_index: int, array: List[int]) -> int:
        """SECURE: Access all array elements to prevent cache timing leaks."""
        result = 0
        for i in range(len(array)):
            mask = constant_time_eq(i, secret_index)
            result |= mask & array[i]
        return result

    @staticmethod
    def constant_time_conditional_swap(condition: int, a: int, b: int) -> Tuple[int, int]:
        """Constant-time conditional swap used in Montgomery ladder."""
        mask = -int(condition & 1)
        temp = mask & (a ^ b)
        a ^= temp
        b ^= temp
        return a, b

    @staticmethod
    def vulnerable_modular_exponentiation(base: int, exponent: int,
                                         modulus: int) -> int:
        """VULNERABLE: Square-and-multiply with secret-dependent branches."""
        result = 1
        base = base % modulus
        while exponent > 0:
            if exponent & 1:
                result = (result * base) % modulus
            exponent >>= 1
            base = (base * base) % modulus
        return result

    @staticmethod
    def constant_time_modular_exponentiation(base: int, exponent: int,
                                             modulus: int) -> int:
        """SECURE: Montgomery ladder for constant-time exponentiation."""
        r0 = 1
        r1 = base % modulus
        exponent_bits = bin(exponent)[2:]
        for bit in exponent_bits:
            bit_value = int(bit)
            temp0 = (r0 * r1) % modulus
            temp1 = (r1 * r1) % modulus
            temp2 = (r0 * r0) % modulus
            mask = -bit_value
            new_r0 = (mask & temp0) | (~mask & temp2)
            new_r1 = (mask & temp1) | (~mask & temp0)
            r0 = new_r0
            r1 = new_r1
        return r0

    @staticmethod
    def constant_time_memory_access(data: np.ndarray, secret_index: int) -> int:
        """Constant-time memory access pattern for arrays."""
        result = 0
        for i in range(len(data)):
            is_target = constant_time_eq(i, secret_index)
            result = constant_time_select(is_target, data[i], result)
        return result

    @staticmethod
    def constant_time_greater_than(a: int, b: int, bit_length: int = 32) -> int:
        """Constant-time comparison: returns 1 if a > b, 0 otherwise."""
        diff = (b - a) & ((1 << bit_length) - 1)
        sign_bit = (diff >> (bit_length - 1)) & 1
        return sign_bit


def constant_time_eq(a: int, b: int) -> int:
    """Constant-time equality test: returns 0xFFFFFFFF if equal, 0 if not."""
    diff = a ^ b
    return ((diff | -diff) >> 31) + 1


def constant_time_select(condition: int, if_true: int, if_false: int) -> int:
    """Constant-time selection between two values."""
    return (condition & if_true) | (~condition & if_false)


class ConstantTimeValidation:
    """Tools to validate constant-time behavior of implementations."""

    @staticmethod
    def measure_timing_variance(func, test_inputs: List[Tuple],
                               iterations: int = 1000) -> float:
        """Measure timing variance of a function across different inputs."""
        import time
        timings = []
        for inputs in test_inputs:
            for _ in range(10):
                func(*inputs)
            times = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                func(*inputs)
                end = time.perf_counter_ns()
                times.append(end - start)
            timings.append(np.median(times))
        return np.std(timings) / (np.mean(timings) + 1e-10)

    @staticmethod
    def validate_constant_time_behavior(vulnerable_func, constant_time_func,
                                      test_cases: List[Tuple]) -> Dict:
        """Compare timing behavior of vulnerable vs constant-time implementations."""
        validator = ConstantTimeValidation()
        vulnerable_variance = validator.measure_timing_variance(vulnerable_func, test_cases)
        constant_variance = validator.measure_timing_variance(constant_time_func, test_cases)
        improvement_factor = vulnerable_variance / (constant_variance + 1e-10)
        return {
            'vulnerable_timing_variance': vulnerable_variance,
            'constant_time_variance': constant_variance,
            'improvement_factor': improvement_factor,
            'is_constant_time': constant_variance < 0.05,
            'test_cases': len(test_cases)
        }


def demonstrate_constant_time_transformations():
    """Demonstrate the constant-time transformations from the paper."""
    ct = ConstantTimeImplementations()
    print("=== Constant-Time Implementation Examples ===\n")
    print("1. Conditional Copy:")
    print("   Vulnerable version uses if-statement")
    print("   Constant-time uses arithmetic operations")
    secret_bit = 1
    result_ct = ct.constant_time_conditional_copy(secret_bit, 42, 17)
    print(f"   Result (bit={secret_bit}): {result_ct}\n")
    print("2. Array Lookup:")
    print("   Vulnerable version directly indexes array")
    print("   Constant-time accesses all elements")
    array = [10, 20, 30, 40, 50]
    secret_idx = 2
    result_ct = ct.constant_time_array_lookup(secret_idx, array)
    print(f"   Result (index={secret_idx}): {result_ct}\n")
    print("3. Conditional Swap (Montgomery Ladder):")
    a, b = 100, 200
    condition = 1
    a_new, b_new = ct.constant_time_conditional_swap(condition, a, b)
    print(f"   Before: a={a}, b={b}")
    print(f"   After (condition={condition}): a={a_new}, b={b_new}\n")
    print("4. Timing Validation:")
    test_cases = [(i, 42, 17) for i in [0, 1]]
    validation = ConstantTimeValidation.validate_constant_time_behavior(
        ct.vulnerable_conditional_copy,
        ct.constant_time_conditional_copy,
        test_cases
    )
    print(f"   Vulnerable timing variance: {validation['vulnerable_timing_variance']:.6f}")
    print(f"   Constant-time variance: {validation['constant_time_variance']:.6f}")
    print(f"   Improvement factor: {validation['improvement_factor']:.2f}x")
    print(f"   Is constant-time: {validation['is_constant_time']}")


if __name__ == "__main__":
    demonstrate_constant_time_transformations()
