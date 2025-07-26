/*
    Author: Thomas Kim
    First Edit: July 21, 2021 - July 26, 2021

    Our primary tool for the implementation of rational number is

    Euclidean Algorithm GCD - std::gcd() - Greatest Common Divisor

*/

#ifndef _CPG_RATIONAL_HPP
#define _CPG_RATIONAL_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_types.hpp"
#include <numeric>
#include <cassert>
#include <cmath>
#include <compare>
#include <optional>

#if defined(INCLUDE_PERMU_COMBI_TABLE) && !defined(PERMU_COMBI_TABLE_INCLUDED)

    #define PERMU_COMBI_TABLE_INCLUDED

    #if defined(PERMU_COMBI_TABLE_NAME)
        #include PERMU_COMBI_TABLE_NAME
    #else
        #include "permu_combi_table.cxx"
    #endif
#endif

namespace cpg::rational_number
{
    #ifdef INCLUDE_PERMU_COMBI_TABLE
        
        extern const std::vector<unsigned long long> factorial_table;
        extern const std::vector<std::vector<unsigned long long>> permutation_table;
        extern const std::vector<std::vector<unsigned long long>> combination_table;

    #endif

    namespace types = cpg::types;

    using real_number_list =
        types::type_container<float, double, long double>;

    template<typename Type>
        concept real_number_c = types::is_in_type_container_c<Type, real_number_list>;
    
    // we will allow all integral types
    using allowed_type_list = 
        types::type_container<char, unsigned char, short, unsigned short,
            int, unsigned int, long, unsigned long,
            long long, unsigned long long>;

    template<typename T>
    concept allowed_type_c
        = types::is_in_type_container_c<T, allowed_type_list>;

    using numerical_type_list = 
        types::type_container<char, unsigned char, short, unsigned short,
            int, unsigned int, long, unsigned long,
            long long, unsigned long long, float, double, long double>;
    
    template<typename Type>
        concept number_c = types::is_in_type_container_c<Type, numerical_type_list>;
     
    // For A = a x G, B = b x G, where G = gcd(A, B)
    // A -> a, B -> b, and returns G, the greatest common divisor,
    // and the sign of b (B reduced) is always positive
    template<allowed_type_c ElementType>
    constexpr ElementType reduce_adjusted(ElementType& A, ElementType& B) noexcept
    {
        try
        {
            auto G = std::gcd(A, B);

            A /= G; B /= G;

            if constexpr(std::is_signed_v<ElementType>)
            {
                // ensure that B is always positive
                if(B < 0)
                {
                    A = -A; B = -B;
                }
            }

            return G;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';

            std::terminate(); // if exception is throw, crash the application asap
                              // such that the client developer can fix his errors
                              // at the time of development
        }
        
        
    }

     // For A = a x G, B = b x G, where G = gcd(A, B)
    // A -> a, B -> b, and returns G, the greatest common divisor
    // the signs of a and b are NOT adjusted
    template<allowed_type_c ElementType>
    constexpr ElementType reduce_simple(ElementType& A, ElementType& B) noexcept
    {
        try
        {
            auto G = std::gcd(A, B);

            A /= G; B /= G; return G;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            std::terminate();
        }
    }

    template<allowed_type_c ElementType>
    class rational
    {
        public:

            ElementType m_num = 0; // numerator
            ElementType m_den = 1; // denominator

            constexpr void reduce() noexcept
            {
                reduce_adjusted(this->m_num, this->m_den);
            }

            constexpr rational() noexcept = default;

            constexpr explicit 
            rational(ElementType numerator) noexcept:
                m_num{numerator} { }

            constexpr rational(ElementType numerator,
                ElementType denominator, bool DoNotReduce = false) noexcept:
                m_num{ numerator }, m_den{ denominator }
                {
                    if(DoNotReduce == false)
                        reduce();
                }
            
            constexpr rational abs() const noexcept
            {
                return { this->m_num > 0 ? this->m_num : -this->m_num,
                    this->m_den, true}; // we do not need to call reduce();
            }

            template<typename Type = ElementType>
            Type mag() const noexcept
            {
                const auto& p = this->m_num;
                const auto& q = this->m_den;

                if constexpr(allowed_type_c<Type>)
                {
                    return (Type)std::round((float) p / q);
                }
                else
                {
                    return (Type)p / q;
                }
            }

            template<typename Type = ElementType>
            Type abs_mag() const noexcept
            {
                const auto& p = this->m_num > 0 ? this->m_num : -this->m_num;
                const auto& q = this->m_den;

                if constexpr(allowed_type_c<Type>)
                {
                    return (Type)std::round((float) p / q);
                }
                else
                {
                    return (Type)p / q;
                }
            }

            constexpr ~rational() { }
            
            // denominator can never be zero
            // the rational number is in valid state
            constexpr bool valid() const noexcept
            {
                return (this->m_den > 0);
            }

            // valid and not zero
            constexpr operator bool() const noexcept
            {
                return valid() && (this->m_num != 0);
            }
            
            // valid and zero
            constexpr bool operator!() const noexcept
            {
                return valid() && (this->m_num == 0);
            }

            // access numerator
            constexpr const ElementType& num() const noexcept
            {
                return this->m_num;
            }

            // access denominator
            constexpr const ElementType& den() const noexcept
            {
                return this->m_den;
            }

            // return old value of the numerator,
            // set a new value to the numerator
            constexpr ElementType num(ElementType new_value, bool DoNotReduce = false) noexcept
            {
                auto old_value = this->m_num;
                this->m_num = new_value;

                if(DoNotReduce == false)
                    this->reduce(); // critically important
                    
                return old_value;
            }

            // return old value of the denominator
            // set a new value to the denominator
            constexpr ElementType den(ElementType new_value, bool DoNotReduce = false) noexcept
            {
                auto old_value = this->m_den;
                this->m_den = new_value;
                
                if(DoNotReduce == false)
                    this->reduce(); // critically important
                
                return old_value;
            }

            // returns old value of the rational number
            // set new value and reduce
            constexpr rational operator()
                (ElementType numerator, ElementType denominator, bool DoNotReduce = false) noexcept
            {
                auto old_value = *this;
                this->m_num = numerator;
                this->m_den = denominator;
                
                if(DoNotReduce==false)
                    this->reduce(); // critically important
                
                return old_value;
            }

            // set new value and reduce
            constexpr void set(ElementType numerator,
                ElementType denominator, bool DoNotReduce = false) noexcept
            {
                this->m_num = numerator;
                this->m_den = denominator;
                
                if(DoNotReduce==false)
                    this->reduce(); // critically important
            }

            // pre-increment operator++()
            // increment the value by 1
            // then return the updated value
            constexpr const rational& operator++() noexcept
            {
                *this += 1; return *this;
            }
            
            // pre-decrement operator--()
            // decrement the value by 1
            // then return the updated value
            constexpr const rational& operator--() noexcept 
            {
                *this -= 1; return *this;
            }

            // post-increment operator++(int)
            // increment the value by 1
            // return the old value
            constexpr rational operator++(int) noexcept
            {
                auto old_value = *this;
                *this += 1;
                return old_value;
            }

            // post-decrement operator--(int) 
            // decrement the value by 1
            // return the old value
            constexpr rational operator--(int) noexcept
            {
                auto old_value = *this;
                *this -= 1; 
                return old_value;
            }

            template<number_c Type = long long>
            constexpr operator Type() const noexcept
            {
                return (Type)this->m_num / this->m_den;
            }

            template<allowed_type_c TargetType = ElementType,
                real_number_c RoundType = long double>
            TargetType round() const noexcept
            {
                return (TargetType)std::round((RoundType)this->m_num / this->m_den);
            }

            friend constexpr std::strong_ordering operator <=>
                (const rational& lhs, const rational& rhs) noexcept
            {
               if(lhs.m_den == rhs.m_den)
               {
                   if(lhs.m_num == rhs.m_num)
                        return std::strong_ordering::equal;
                   else if(lhs.m_num < rhs.m_num)
                        return std::strong_ordering::less;
                   else
                        return std::strong_ordering::greater;
               }
               else
               {
                    // we subtract two floats
                    auto diff = (float)lhs - (float)rhs;

                    if(diff < 0)
                        return std::strong_ordering::less;
                    else if (diff > 0)
                        return std::strong_ordering::greater;
                    else
                    {
                        // this part will never get executed
                        return std::strong_ordering::equal;
                    }
               }
            }

            friend constexpr std::strong_ordering operator <=>
                (const rational& lhs, ElementType rhs) noexcept
            {
                return lhs <=> rational{rhs};
            }

            
            friend constexpr std::strong_ordering operator <=>
                (ElementType lhs, const rational& rhs) noexcept
            {
                return rational{lhs} <=> rhs;
            }

            friend constexpr bool operator==
            (const rational& lhs, const rational& rhs) noexcept 
            {
                if( lhs.m_den == rhs.m_den && lhs.m_num == rhs.m_num) 
                    return true;
                else 
                    return false;
            }

            friend constexpr bool operator==
            (const rational& lhs, ElementType rhs) noexcept 
            {
                return lhs == rational{rhs};
            }

            friend constexpr bool operator==
            (ElementType lhs, const rational& rhs) noexcept 
            {
                return rational{lhs} == rhs;
            }

            friend constexpr bool operator!=
            (const rational& lhs, const rational& rhs) noexcept
            {
               return !(lhs == rhs);
            }

            friend constexpr bool operator!=
            (const rational& lhs, ElementType rhs) noexcept
            {
                return lhs != rational{rhs};
            }

            friend constexpr bool operator!=
            (ElementType lhs, const rational& rhs) noexcept
            {
                return rational{lhs} != rhs;
            }


            ////////////// Multiplications and Divisions ///////////////////////

            // multiplicative inverse
            constexpr rational reciprocal() const noexcept
            {
                // return { this->m_den, this->m_num };

                if constexpr(std::is_signed_v<ElementType>)
                {
                    if(this->m_num < 0)
                        return { -this->m_den, -this->m_num, true };
                    else
                        return { this->m_den, this->m_num, true };

                }
                else // unsigned integral types,
                     // such as unsigned int, unsigned long long
                {
                    // we don't need to reduce, because 
                    // m_den, n_num are already reduced
                    return { this->m_den, this->m_num, true };
                }
            }

            constexpr rational& operator *=(const rational& rhs) noexcept
            {
                /*
                     this->m_num    rhs.m_num      rhs.m_num       this->m_num
                    ------------ x ----------- = ------------ x ----------------
                     this->m_den    rhs.m_den     this->m_den     rhs.m_den

                */

               rational L{ rhs.m_num, this->m_den};
               rational R{ this->m_num,  rhs.m_den};

               *this = rational{ L.m_num * R.m_num, L.m_den * R.m_den };

               return *this;
            }
          
            constexpr rational& operator /=(const rational& rhs) noexcept
            {
                *this *= rhs.reciprocal(); return *this;
            }

            constexpr rational& operator *=(ElementType rhs) noexcept
            {
                /*  this->m_num                            rhs
                    ----------- x rhs = this->m_num x -------------
                    this->m_den                         this->m_den
                */

               // auto num = this->m_num;
               rational R{rhs, this->m_den};

               *this = rational{ this->m_num * R.m_num, R.m_den };

               return *this;
            }

            friend constexpr rational
            operator*(ElementType lhs, rational rhs) noexcept
            {
                rhs *= lhs; return rhs;
            }

            friend constexpr rational
            operator*(rational lhs, ElementType rhs) noexcept
            {
                lhs *= rhs; return lhs;
            }

            constexpr rational& operator /=(ElementType rhs) noexcept
            {
                /*  this->m_num      1      this->m_num        1
                    ----------- x ------ = ------------ x -----------
                    this->m_den     rhs        rhs          this->m_den
                */

               
               rational L{this->m_num, rhs};

               *this = rational{ L.m_num,  L.m_den * this->m_den };

               return *this;
            }

            friend constexpr rational
            operator/(ElementType lhs, const rational& rhs) noexcept
            {
                auto R = rhs.reciprocal();
                R *= lhs; return R;
            }

            friend constexpr rational
            operator/(const rational &lhs, ElementType rhs) noexcept
            {
                auto [A, P] = lhs; // A is numerator, P is denominator

                /*       A          1             1           A
                    ---------- * --------- =  ---------- x -----------
                         P         rhs              P        rhs
                */

               reduce_simple(A, rhs); // (A / rhs)

               return { A, P * rhs };
            }

            friend constexpr rational
            operator*(rational lhs, const rational& rhs) noexcept
            {
                lhs *= rhs; return lhs;
            }

            friend constexpr rational
            operator/(rational lhs, const rational& rhs) noexcept
            {
                lhs /= rhs; return lhs;
            }

            /////////////////////////// Additions and Subtractions ////////////////

            // additive inverse
            constexpr rational inverse() const noexcept
            {
                return { -this->m_num, this->m_den, true };
            }

            // additvie inverse
            constexpr rational operator-() const noexcept
            {
                return { -this->m_num, this->m_den, true };
            }
            
            constexpr rational& operator +=(const rational& rhs) noexcept
            {
                auto [A, P] = *this; // IMPORTANT - P, Q are denominators
                auto [B, Q] = rhs;   //             A, B are numerators

                /*
                      A       B          A           B
                    ----- + ----- = ---------- + --------, G = gcd(P, Q)
                      P       Q        p x G       q x G

                       A            B        q x A          p x B 
                    --------- + -------- = ---------- + ------------
                      p x G       q x G    q x p x G       p x q x G


                      q x A  +  p x B 
                    -------------------
                         q x p x G       
                */

               // P -> p, Q -> q
                auto G = rational_number::reduce_simple(P, Q);
                *this = rational{ Q * A  +  P * B,  Q * P * G };
                return *this;
            }

            constexpr rational& operator -=(const rational& rhs) noexcept
            {
                auto [A, P] = *this; // IMPORTANT - P, Q are denominators
                auto [B, Q] = rhs;   //             A, B are numerators

                /*
                      A       B          A           B
                    ----- - ----- = ---------- - --------, G = gcd(P, Q)
                      P       Q        p x G       q x G

                       A            B        q x A          p x B 
                    --------- - -------- = ---------- - ------------
                      p x G       q x G    q x p x G       p x q x G


                      q x A  -  p x B 
                    -------------------
                         q x p x G       
                */

               // P -> p, Q -> q
                auto G = rational_number::reduce_simple(P, Q);
                *this = rational{ Q * A  -  P * B,  Q * P * G };
                return *this;
            }

            friend constexpr rational
                operator+(rational lhs, const rational& rhs) noexcept
            {
                lhs += rhs; return lhs;
            }

            friend constexpr rational
                operator-(rational lhs, const rational& rhs) noexcept
            {
                lhs -= rhs; return lhs;
            }

            constexpr rational& operator +=(ElementType rhs) noexcept
            {
                auto [A, P] = *this; // IMPORTANT - P is denominator
                                     //             A is numerator
                
                /*
                      A      rhs      A        P x rhs
                    ----- + ----- = ------- + --------
                      P       1       P           P

                      A  +  P x rhs 
                    -------------------
                         P       
                */

                *this = rational{ A  +  P * rhs,  P};
                return *this;
            }

            constexpr rational& operator -=(ElementType rhs) noexcept
            {
                auto [A, P] = *this; // IMPORTANT - P is denominator
                                     //             A is numerator
                
                /*
                      A      rhs      A        P x rhs
                    ----- - ----- = ------- - --------
                      P       1       P           P

                      A  -  P x rhs 
                    -------------------
                         P       
                */

                *this = rational{ A  -  P * rhs,  P};
                return *this;
            }

            friend constexpr rational
                operator +(ElementType lhs, rational rhs) noexcept
            {
                rhs +=  lhs; return rhs;  
            }

            friend constexpr rational
                operator -(ElementType lhs, const rational& rhs) noexcept
            {
               auto& [B, Q] = rhs; // Q is denominator, B is numerator

               /*
                      lhs       B        lhs * Q - B
                    ----- -  ------- = --------------
                      1         Q            Q
               */   

               return { lhs * Q - B, Q };
            }

            friend constexpr rational
                operator+(rational lhs, ElementType rhs) noexcept
            {
                lhs += rhs; return lhs;   
            }

            friend constexpr rational
                operator - (rational lhs, ElementType rhs) noexcept
            {
                lhs -= rhs; return lhs;   
            }

            template<typename CharType>
            friend std::basic_ostream<CharType>& operator<<(
                std::basic_ostream<CharType>& os, const rational& r) noexcept
            {
                if constexpr(std::is_same_v<ElementType, char>)
                {
                    os << "( " << (short)r.m_num << ", " << (short)r.m_den <<" )";
                }
                else if constexpr(std::is_same_v<ElementType, unsigned char> )
                {
                    os << "( " << (unsigned short)r.m_num << ", " 
                        << (unsigned short)r.m_den <<" )";
                }
                else
                {
                    os << "( " << r.m_num << ", " << r.m_den <<" )";
                }
                return os;
            }
    };
    // end of class rational

    #ifdef INCLUDE_PERMU_COMBI_TABLE
        
        auto factorial(unsigned long long n) noexcept
        {
            std::optional<unsigned long long> result;

            if(n < factorial_table.size() )
                result = factorial_table[n];

            return result;
        }
        // end of factorial()

        auto nPr(unsigned long long n, unsigned long long r) noexcept
        {
            std::optional<unsigned long long> result;

            if(r == 0)
            {
                result = 1; return result;
            }
            else if (r == 1)
            {
                result = n;  return result;
            }
            else if( n < permutation_table.size())
            {
                if(r < permutation_table[n].size())
                    result = permutation_table[n][r];
                
                return result;
            }
            else
            {
                unsigned long long old_value = 0;
                unsigned long long new_value = 1;
                
                bool success = true;

                for(unsigned long long k = 0; k < r; ++k )
                {
                    old_value = new_value;
                    new_value *= (n-k);

                    if(old_value != new_value /(n-k))
                    {
                        success = false; break;
                    }
                }

                if(success) result = new_value;

                return result;
            }
        }
        // end of nPr()

        auto nCr(unsigned long long n, unsigned long long r) noexcept
        {
            std::optional<unsigned long long> result;

            if(r == 0 || r == n) 
            {
                result = 1; return result; 
            }

            if ( r > (n-r) ) r = n-r;
            
            if( n < combination_table.size())
            {
                if ( r > (n-r) ) r = n-r;

                if(r < combination_table[n].size())
                    result = combination_table[n][r];
                
                return result;
            }
            else
            {
                /*                          nPr         (n - 0) x (n - 1) x .... x (n - (r-1) )
                nCr * r! = nPr => nCr = ---------- = -------------------------------------------
                                            r!          (r-0)  x (r - 1) x .....x (r - (r-1) )           

                nCr, r = 0  => 1, r = n => 1
                nCr = nCn-r, r = min{ r, n-r }
                */

                rational<unsigned long long> old_value;
                rational<unsigned long long> new_value{1};

                bool success = true;

                for(unsigned long long k = 0; k < r; ++k)
                {
                    auto value = rational<unsigned long long>{n - k, r - k};

                    old_value = new_value;
                    new_value *= value;

                    if(old_value != (new_value / value))
                    {
                        success = false; break;
                    }
                }

                if(success)
                    result = (unsigned long long)new_value;

                return result;
            }
        }
        // end of nCr()
    #else
        auto factorial(unsigned long long n) noexcept
        {
            std::optional<unsigned long long> result;

            unsigned long long old_value;
            unsigned long long new_value = 1;
            bool success = true;

            for(unsigned long long k = 1; k <= n; ++k)
            {
                old_value = new_value; new_value *= k;

                if(old_value != new_value / k) 
                {
                    success = false; break;
                }
            }

            if(success) result = new_value;

            return result;
        }
        // end of factorial()

        auto nPr(unsigned long long n, unsigned long long r = 1) noexcept
        {
            std::optional<unsigned long long> result;

            unsigned long long old_value;
            unsigned long long new_value = 1;
            bool success = true;

            for(unsigned long long k = 0; k < r; ++k)
            {
                old_value = new_value;
                new_value *= (n-k);

                if(old_value != (new_value / (n-k)))
                {
                    success = false; break;
                }
            }

            if(success) result = new_value;

            return result;
        }
        // end of nPr()

        auto nCr(unsigned long long n, unsigned long long r)  noexcept
        {
            /*                             nPr         (n - 0) x (n - 1) x .... x (n - (r-1) )
                nCr * r! = nPr => nCr = ---------- = ----------------------------------
                                            r!          (r-0)  x (r - 1) x .....x (r - (r-1) )           


                nCr, r = 0  => 1, r = n => 1
                nCr = nCn-r, r = min{ r, n-r }
            */

            std::optional<unsigned long long> result;

            if(r == 0 || r == n)
            {
                result = 1; return result;
            }

            if ( r > (n-r) ) r = n-r;
            
            rational<unsigned long long> old_value;
            rational<unsigned long long> new_value{1};

            bool success = true;

            for(unsigned long long k = 0; k < r; ++k)
            {
                auto value = rational<unsigned long long>{n - k, r - k};

                old_value = new_value;
                new_value *= value;

                if(old_value != (new_value / value))
                {
                    success = false; break;
                }
            }

            if(success)
                result = (unsigned long long)new_value;

            return result;
        }
        // end of nCr()

    #endif // end of INCLUDE_PERMU_COMBI_TABLE
}
// end of namespace rational_number

#endif // end of file _CPG_RATIONAL_HPP