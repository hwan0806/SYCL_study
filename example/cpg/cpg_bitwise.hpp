#ifndef _CPG_BITWISE_HPP
#define _CPG_BITWISE_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include <iostream>
#include <type_traits>
#include <limits>
#include <string>
#include <sstream>
#include <iomanip>  
#include <cstddef> // for std::byte
#include <bitset>
#include <cmath>
#include <concepts>
#include <bit>

/*
	105 - C++ Casting 3 - C++20 새로운 비트 연산 라이브러리, bit, concepts, algorithm
	https://www.youtube.com/watch?v=AXSGZDVDtdc&list=PLsIvhalfft1EtaiJJRmWKUGG0W1hyUKiw&index=104
	
*/

namespace cpg::bitwise
{
	namespace hidden
	{
		template<typename T> struct st_is_character
		{
			static const bool value = false;
		};

		template<> struct st_is_character<char>
		{
			static const bool value = true;
		};

		template<> struct st_is_character<unsigned char>
		{
			static const bool value = true;
		};

		template<> struct st_is_character<signed char>
		{
			static const bool value = true;
		};

		template<> struct st_is_character<wchar_t>
		{
			static const bool value = true;
		};
	}

	template<typename T>
	constexpr bool is_char_v = hidden::st_is_character<std::remove_cvref_t<T>>::value;

	template<typename T> constexpr T limits_max = std::numeric_limits<T>::max();
	template<typename T> constexpr T limits_min = std::numeric_limits<T>::min();
	
	// count bits in the mask
	// mask ( 7)      : 0000 0111,  should return 3
	// mask (-1)      : 1111 1111,  should return 8
	// mask (0x80)    : 1000 0000,  should return 1
	// mask (0)       : 0000 0000,  should return 0
	// mask (1)       : 0000 0001,  should return 1

	template<typename T>
	using enable_if_integral = std::enable_if_t<std::is_integral<T>::value>;

	template<typename T, typename = enable_if_integral<T>>
	using unsigned_t = std::make_unsigned_t<T>;

	template<typename T, typename = enable_if_integral<T>>
	using signed_t = std::make_signed_t<T>;

	template<typename S, typename T>
	using common_t = std::common_type_t<S, T>;

	template<typename S, typename T,
		typename = enable_if_integral<S>,
		typename = enable_if_integral<T>>
		using signed_common_t = signed_t<common_t<S, T>>;

	template<typename S, typename T,
		typename = enable_if_integral<S>,
		typename = enable_if_integral<T>>
		using unsigned_common_t = unsigned_t<common_t<S, T>>;

	template<typename T>
	constexpr int sizeof_bits = sizeof(T) * 8;

	template<typename T>
	constexpr unsigned_t<T> high_bit_mask
		= unsigned_t<T>(1) << (sizeof_bits<T>-1);

	template<std::integral T>
	int count_set_bits(T bits)
	{
		unsigned_t<T> mask = bits;

		int count = 0;

		for (; mask; mask >>= 1)
			count += (int)(mask & 1);

		return count;
	}
			
	template<std::integral T>
	std::string to_bits(T bits)
	{
		unsigned_t<T> mask = bits;
		int count = 0;
		
		std::ostringstream os;

		for (int pos = sizeof_bits<T>; pos; mask <<= 1, --pos)
		{
			if (!os.str().empty() && (pos % 4 == 0))
				os << ' ';
				
			os << ((mask & high_bit_mask<T>) ? '1' : '0');
		}

		return os.str();
	}

	template<std::integral T, std::size_t N>
	std::string to_bits_reverse(T(&bits)[N])
	{
		std::ostringstream os;

		for(int n = (int)N-1; n > 0 ; --n)
		{
			os << to_bits(bits[n]) <<" | ";
		}

		os << to_bits(bits[0]);

		return os.str();
	}

	template<std::integral T, std::size_t N>
	std::string to_bits(T(&bits)[N])
	{
		std::ostringstream os;

		for(int n = 0; n < (int)N - 1; ++n)
		{
			os << to_bits(bits[n]) <<" | ";
		}

		os << to_bits(bits[N-1]);

		return os.str();
	}

	template<std::integral T>
	std::string to_hex(T v)
	{
		std::string str;
					
		if (is_char_v<T>)
		{
			std::ostringstream os;

			os << std::setfill('0') << std::setw(sizeof(T) * 2)
				<< std::uppercase << std::hex << (short)v;

			str = os.str();

			if (str.size()==4)
			{
				if (v > 0)
				{
					str.pop_back();
					str.pop_back();
				}
				else
				{
					std::string ss;

					ss.push_back(str[2]);
					ss.push_back(str[3]);
					
					str = ss;
				}
			}
		}
		else
		{
			std::ostringstream os;

			os << std::setfill('0') << std::setw(sizeof(T) * 2)
				<< std::uppercase << std::hex <<v;
			
			str = os.str();
		}

		return str;
	}

	template<std::integral T, std::size_t N>
	std::string to_hex_reverse(T(&v)[N])
	{
		std::ostringstream os;

		for(int n = (int)N-1; n > 0 ; --n)
		{
			os << to_hex(v[n]) <<" | ";
		}

		os << to_hex(v[0]);

		return os.str();
	}

	template<std::integral T, std::size_t N>
	std::string to_hex(T(&v)[N])
	{
		std::ostringstream os;

		for(int n = 0; n < (int)N - 1 ; ++n)
		{
			os << to_hex(v[n]) <<" | ";
		}

		os << to_hex(v[N-1]);

		return os.str();
	}

	template<std::integral T>
	std::string to_dec(T v)
	{
		std::ostringstream os;
		
		if (is_char_v<T>)
			os << (short)v;
		else
			os << v;

		return os.str();
	}

	template<std::integral T, std::size_t N>
	std::string to_dec_reverse(T(&v)[N])
	{
		std::ostringstream os;

		for(int n = (int)N-1; n > 0; --n)
		{
			os << to_dec(v[n]) <<" | ";
		}

		os << to_dec(v[0]);

		return os.str();
	}

	template<std::integral T, std::size_t N>
	std::string to_dec(T(&v)[N])
	{
		std::ostringstream os;

		for(int n = 0; n < (int)N - 1; ++n)
		{
			os << to_dec(v[n]) <<" | ";
		}

		os << to_dec(v[N-1]);

		return os.str();
	}

	template<std::integral T>
	int field_with(T v)
	{
		std::ostringstream os;

		if (is_char_v<T>)
			os << (short)v;
		else
			os << v;

		return (int) os.str().size();
	}
	
	template<std::integral T>
	int numeric_width()
	{
		int a = field_with(std::numeric_limits<T>::max());
		int b = field_with(std::numeric_limits<T>::min());

		return a > b ? a : b;
	}

	template<std::integral T>
	std::string to_dec_width(T v)
	{
		std::ostringstream os;
		
		int max_field = numeric_width<T>();

		if (is_char_v<T>)
			os << std::setw(max_field)<<(short)v;
		else
			os << std::setw(max_field)<<v;

		return os.str();
	}

	template<std::integral T>
	std::string numeric_base(T v)
	{
		std::ostringstream os;

		os << to_dec_width(v) << " (" << to_hex<T>(v) << "): " << to_bits<T>(v);

		return os.str();
	}

	template<std::integral T>
	std::string numeric_type_info()
	{
		std::ostringstream os;

		auto minimum = std::numeric_limits<T>::min();
		auto maximum = std::numeric_limits<T>::max();

		os << "Type name: " << Cpg_GetTypeName(T)
			<< ",\tByte size: " << sizeof(T) << ",\tBit count: " << sizeof_bits<T>
			<< "\nMinimum: "<< numeric_base(minimum)
		    << "\nMaximum: "<< numeric_base(maximum);

		return os.str();
	}

	template<typename = void>
	std::string integral_type_info()
	{
		std::ostringstream os;

		os << numeric_type_info<char>() << "\n\n";
		os << numeric_type_info<unsigned char>() << "\n\n";
		os << numeric_type_info<short>() << "\n\n";
		os << numeric_type_info<unsigned short>() << "\n\n";
		os << numeric_type_info<int>() << "\n\n";
		os << numeric_type_info<unsigned int>() << "\n\n";
		os << numeric_type_info<long>() << "\n\n";
		os << numeric_type_info<unsigned long>() << "\n\n";
		os << numeric_type_info<long long>() << "\n\n";
		os << numeric_type_info<unsigned long long>() << "\n";

		return os.str();
	}

	template<std::integral T>
	std::string twos_complement(T c)
	{
		std::ostringstream os;
		os << numeric_type_info<T>() << "\n";

		T c1 = ~c;   // 1's complement of s1
		T c2 = ~c + 1; // 2's complement of s1
		T c3 = c + c2; // c + (~c+1)
		os << "original value  :      c     = " << numeric_base(c) <<  "\n";
		os << "1's complement  :     ~c     = " << numeric_base(c1) << "\n";
		os << "2's complement  :   ~c + 1   = " << numeric_base(c2) << "\n";
		os << "complement check: c + (~c+1) = " << numeric_base(c3) << "\n";

		return os.str();
	}

	template<std::unsigned_integral T>
	struct bit_pattern
	{
		constexpr static size_t byte_size = sizeof(T);
		constexpr static T min_limit = std::numeric_limits<T>::min();
		constexpr static T max_limit = std::numeric_limits<T>::max();

		// on some machine, bit count of a byte is not 8 bits, but 16 bits
		// but I will ignore it, and assume 1 byte represents 8 bits
		constexpr static size_t bit_size = sizeof(T) * 8;

		union
		{
			// n for number or numerical interpretation
			T n{};

			// b for binary digits - b is array of type std::byte
			std::byte b[byte_size];
		};

		// most significant byte
		const std::byte& msb() const noexcept
		{
			return b[byte_size - 1];
		}
		
		// most significant byte
		std::byte& msb() noexcept
		{
			return b[byte_size - 1];
		}

		// least significant byte
		const std::byte& lsb() const noexcept
		{
			return b[0];
		}
		
		// least significant byte
		std::byte& lsb() noexcept
		{
			return b[0];
		}

		template<typename S>
			requires std::floating_point<S> || std::integral<S>
		bit_pattern& operator=(S s) noexcept
		{
			if constexpr(std::floating_point<S>)
			{
				// s is a floating point type
				this->n = (T) std::round(s); // 1. round(), then 2. typecast S to T
			}
			else
			{
				// s is an integral type
				this->n = (T)s; // typecast S to T
			}
			
			return *this;
		}

		// we allow both integral and real numbers to be assigned by this class
		template<typename S>
			requires std::floating_point<S> || std::integral<S>
		operator S() const noexcept
		{
			return (S)this->n;
		}
		
	}; // end of class bit_pattern
}

#define Tpf_SignedCommonType(a, b) tpf::bitwise::signed_common_t<decltype(a), decltype(b)>
#define Tpf_UnignedCommonType(a, b) tpf::bitwise::unsigned_common_t<decltype(a), decltype(b)>

#endif // end of file _CPG_BITWISE_HPP
