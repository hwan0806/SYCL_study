/*
    Author: Thomas Kim
    First Edit: Dec. 03, 2021
    Second Edit: Dec. 05, 2021
    Third Edit: Dec. 09, 2021 - safe binary operation
    Fourth Edit: Dec. 12, 2021 - std::array - std::vector operation
*/

#ifndef _CGP_VECTOR_OPERATIONS_HPP
#define _CGP_VECTOR_OPERATIONS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_types.hpp"

namespace std::inline stl_extensions
{
    namespace cgt = cpg::types;
           
    template<cgt::vector_c LeftType, cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator+(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        assert(A.size() == B.size());

        using element_t_a = cgt::first_type_t<LeftType>; 
        using element_t_b = cgt::first_type_t<RightType>;
       
        std::size_t Size = A.size();

        if constexpr( std::same_as<element_t_a, element_t_b> &&
                        std::is_rvalue_reference_v<decltype(A) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = cgt::sbo(A[i]) + cgt::sbo(B[i]);

            return std::move(A);
        }
        else if constexpr( std::same_as<element_t_a, element_t_b> &&
                            std::is_rvalue_reference_v<decltype(B) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = cgt::sbo(A[i]) + cgt::sbo(B[i]);

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A[i]) + cgt::sbo(B[i]);

            return C; 
        }
    }

    template<cgt::vector_c LeftType, cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator-(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        assert(A.size() == B.size());

        using element_t_a = cgt::first_type_t<LeftType>; 
        using element_t_b = cgt::first_type_t<RightType>;
       
        std::size_t Size = A.size();

        if constexpr( std::same_as<element_t_a, element_t_b> &&
                        std::is_rvalue_reference_v<decltype(A) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = cgt::sbo(A[i]) - cgt::sbo(B[i]);

            return std::move(A);
        }
        else if constexpr( std::same_as<element_t_a, element_t_b> &&
                            std::is_rvalue_reference_v<decltype(B) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = cgt::sbo(A[i]) - cgt::sbo(B[i]);

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A[i]) - cgt::sbo(B[i]);

            return C; 
        }
    }

    template<cgt::vector_c LeftType, cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator*(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        assert(A.size() == B.size());

        using element_t_a = cgt::first_type_t<LeftType>; 
        using element_t_b = cgt::first_type_t<RightType>;
       
        std::size_t Size = A.size();

        if constexpr( std::same_as<element_t_a, element_t_b> &&
                        std::is_rvalue_reference_v<decltype(A) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = cgt::sbo(A[i]) * cgt::sbo(B[i]);

            return std::move(A);
        }
        else if constexpr( std::same_as<element_t_a, element_t_b> &&
                            std::is_rvalue_reference_v<decltype(B) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = cgt::sbo(A[i]) * cgt::sbo(B[i]);

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A[i]) * cgt::sbo(B[i]);

            return C; 
        }
    }

    template<cgt::vector_c LeftType, cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator/(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        assert(A.size() == B.size());

        using element_t_a = cgt::first_type_t<LeftType>; 
        using element_t_b = cgt::first_type_t<RightType>;
       
        std::size_t Size = A.size();

        if constexpr( std::same_as<element_t_a, element_t_b> &&
                        std::is_rvalue_reference_v<decltype(A) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = cgt::sbo(A[i]) / cgt::sbo(B[i]);
            
            return std::move(A);
        }
        else if constexpr( std::same_as<element_t_a, element_t_b> &&
                            std::is_rvalue_reference_v<decltype(B) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = cgt::sbo(A[i]) / cgt::sbo(B[i]);

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A[i]) / cgt::sbo(B[i]);

            return C; 
        }
    }

    // vector cross product
    template<cgt::vector_c LeftType, cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType> 
        && (cgt::std_array_c<cgt::first_type_t<LeftType>>)
        && (cgt::std_array_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator%(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        assert(A.size() == B.size());

        using element_t_a = cgt::first_type_t<LeftType>; 
        using element_t_b = cgt::first_type_t<RightType>;
       
        std::size_t Size = A.size();

        if constexpr( std::same_as<element_t_a, element_t_b> &&
                        std::is_rvalue_reference_v<decltype(A) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = A[i] % B[i];

            return std::move(A);
        }
        else if constexpr( std::same_as<element_t_a, element_t_b> &&
                            std::is_rvalue_reference_v<decltype(B) >)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = A[i] % B[i];

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = A[i] % B[i];

            return C; 
        }
    }

    // vector inner product
    template<cgt::vector_c LeftType, cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType> 
        && (cgt::std_array_c<cgt::first_type_t<LeftType>>)
        && (cgt::std_array_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator&(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        assert(A.size() == B.size());

        using element_t_a = cgt::first_type_t<LeftType>; 
        using element_t_b = cgt::first_type_t<RightType>;
       
        std::size_t Size = A.size();

        using c_t = cgt::common_vector_t<LeftType, RightType>;
        using element_t =std::tuple_element_t<0, cgt::first_type_t<c_t>>;
        using vector_t = std::vector<element_t>;

        vector_t R(A.size());

        for(std::size_t i=0; i < A.size(); ++i)
            R[i] = A[i] & B[i];

        return R;
    }

   /////////////////////////////////////////////////////////////////////////
    template<cgt::vector_c LeftType, cgt::std_array_flat_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::std_array_c<cgt::first_type_t<LeftType>>)
           && (cgt::numerical_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator+(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = A.size();

        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i) A[i] = A[i] + B;

            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A[i] + B;

            return C; 
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::vector_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::numerical_c<cgt::first_type_t<LeftType>>)
           && (cgt::std_array_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator+(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = B.size();

        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i) B[i] = A + B[i];

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A + B[i];

            return C; 
        }
    }
 
   /////////////////////////////////////////////////////////////////////////
    template<cgt::vector_c LeftType, cgt::std_array_flat_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::std_array_c<cgt::first_type_t<LeftType>>)
           && (cgt::numerical_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator-(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = A.size();

        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i) A[i] = A[i] - B;

            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A[i] - B;

            return C; 
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::vector_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::numerical_c<cgt::first_type_t<LeftType>>)
           && (cgt::std_array_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator-(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = B.size();

        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i) B[i] = A - B[i];

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A - B[i];

            return C; 
        }
    }
 
    /////////////////////////////////////////////////////////////////////////
    template<cgt::vector_c LeftType, cgt::std_array_flat_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::std_array_c<cgt::first_type_t<LeftType>>)
           && (cgt::numerical_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator*(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = A.size();

        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i) A[i] = A[i] * B;

            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A[i] * B;

            return C; 
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::vector_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::numerical_c<cgt::first_type_t<LeftType>>)
           && (cgt::std_array_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator*(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {   
        std::size_t Size = B.size();

        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i) B[i] = A * B[i];

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A * B[i];

            return C; 
        }
    }
  
    /////////////////////////////////////////////////////////////////////////
    template<cgt::vector_c LeftType, cgt::std_array_flat_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::std_array_c<cgt::first_type_t<LeftType>>)
           && (cgt::numerical_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator/(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = A.size();

        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i) A[i] = A[i] / B;

            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A[i] / B;

            return C; 
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::vector_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::numerical_c<cgt::first_type_t<LeftType>>)
           && (cgt::std_array_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator/(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = B.size();

        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i) B[i] = A / B[i];

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A / B[i];

            return C; 
        }
    }

    /////////////////////////////////////////////////////////////////////////
    template<cgt::vector_c LeftType, cgt::std_array_flat_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::std_array_c<cgt::first_type_t<LeftType>>)
           && (cgt::numerical_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator%(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = A.size();

        // std::cout << "cgt::first_type_t<RightType> ' type : " 
        //     >> cgt::first_type_t<RightType>{} << std::endl;

        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i) A[i] = A[i] % B;

            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A[i] % B;

            return C; 
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::vector_c RightType>
         requires cgt::common_vector_c<LeftType, RightType> 
           && (cgt::numerical_c<cgt::first_type_t<LeftType>>)
           && (cgt::std_array_c<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator%(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        std::size_t Size = B.size();

        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i) B[i] = A % B[i];

            return std::move(B);
        }
        else
        {
            using c_t = cgt::common_vector_t<LeftType, RightType>; c_t C(Size);

            for(std::size_t i=0; i < Size; ++i) C[i] = A % B[i];

            return C; 
        }
    }
 
    /////////////////////////////////////////////////////////////////////////
    template<cgt::vector_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::std_array_c<cgt::first_type_t<LeftType>>
        && cgt::numerical_c<cgt::first_type_t<RightType>> 
        && (std::tuple_size_v<cgt::first_type_t<LeftType>> 
            == std::tuple_size_v<std::remove_cvref_t<RightType>>)
    constexpr decltype(auto) operator&(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using element_t_a = cgt::first_type_t<cgt::first_type_t<LeftType>>;
        using element_t_b = cgt::first_type_t<RightType>;
       
        std::size_t Size = A.size();

        using element_t = cgt::common_vector_t<element_t_a, element_t_b>;
        using vector_t = std::vector<element_t>;

        vector_t R(Size);

        for(std::size_t i=0; i < Size; ++i) R[i] = A[i] & B;

        return R;
    }

    template<cgt::std_array_flat_c LeftType, cgt::vector_c RightType>
        requires cgt::numerical_c<cgt::first_type_t<LeftType>>
        && cgt::std_array_c<cgt::first_type_t<RightType>> 
        && (std::tuple_size_v<std::remove_cvref_t<LeftType>> 
            == std::tuple_size_v<cgt::first_type_t<RightType>>)
    constexpr decltype(auto) operator&(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using element_t_a = cgt::first_type_t<LeftType>;
        using element_t_b = cgt::first_type_t<cgt::first_type_t<RightType>>;
       
        std::size_t Size = B.size();

        using element_t = cgt::common_vector_t<element_t_a, element_t_b>;
        using vector_t = std::vector<element_t>;

        vector_t R(Size);

        for(std::size_t i=0; i < Size; ++i) R[i] = A & B[i];

        return R;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    template<cgt::vector_c LeftType, cgt::numerical_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator+(LeftType&& A, RightType B) noexcept(!cpg::bDetectOverFlow)
    {
        using c_t = cgt::common_vector_t<LeftType, RightType>;

        std::size_t Size = A.size();

       if constexpr( std::same_as<c_t, std::remove_cvref_t<LeftType>> &&
                        std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = cgt::sbo(A[i]) + cgt::sbo(B); 

            return std::move(A);
        }
        else
        {
            c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A[i]) + cgt::sbo(B);

            return C; 
        }
    }

    template<cgt::numerical_c LeftType,  cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator+(LeftType A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using c_t = cgt::common_vector_t<LeftType, RightType>;

        std::size_t Size = B.size();

       if constexpr(std::same_as<c_t, std::remove_cvref_t<RightType>> &&
                        std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = cgt::sbo(A) + cgt::sbo(B[i]); 

            return std::move(B);
        }
        else
        {
            c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A) + cgt::sbo(B[i]);

            return C; 
        }
    }

    template<cgt::vector_c LeftType, cgt::numerical_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator-(LeftType&& A, RightType B) noexcept(!cpg::bDetectOverFlow)
    {
        using c_t = cgt::common_vector_t<LeftType, RightType>;

        std::size_t Size = A.size();

       if constexpr( std::same_as<c_t, std::remove_cvref_t<LeftType>> &&
                        std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = cgt::sbo(A[i]) - cgt::sbo(B); 

            return std::move(A);
        }
        else
        {
            c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A[i]) - cgt::sbo(B);

            return C; 
        }
    }

    template<cgt::numerical_c LeftType,  cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator-(LeftType A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using c_t = cgt::common_vector_t<LeftType, RightType>;

        std::size_t Size = B.size();

       if constexpr(std::same_as<c_t, std::remove_cvref_t<RightType>> &&
                        std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = cgt::sbo(A) - cgt::sbo(B[i]); 

            return std::move(B);
        }
        else
        {
            c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A) - cgt::sbo(B[i]);

            return C; 
        }
    }

    template<cgt::vector_c LeftType, cgt::numerical_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator*(LeftType&& A, RightType B) noexcept(!cpg::bDetectOverFlow)
    {
        using c_t = cgt::common_vector_t<LeftType, RightType>;

        std::size_t Size = A.size();

       if constexpr( std::same_as<c_t, std::remove_cvref_t<LeftType>> &&
                        std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = cgt::sbo(A[i]) * cgt::sbo(B); 

            return std::move(A);
        }
        else
        {
            c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A[i]) * cgt::sbo(B);

            return C; 
        }
    }

    template<cgt::numerical_c LeftType,  cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator*(LeftType A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using c_t = cgt::common_vector_t<LeftType, RightType>;

        std::size_t Size = B.size();

       if constexpr(std::same_as<c_t, std::remove_cvref_t<RightType>> &&
                        std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = cgt::sbo(A) * cgt::sbo(B[i]); 

            return std::move(B);
        }
        else
        {
            c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A) * cgt::sbo(B[i]);

            return C; 
        }
    }

    template<cgt::vector_c LeftType, cgt::numerical_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator/(LeftType&& A, RightType B) noexcept(!cpg::bDetectOverFlow)
    {
        using c_t = cgt::common_vector_t<LeftType, RightType>;

        std::size_t Size = A.size();

       if constexpr( std::same_as<c_t, std::remove_cvref_t<LeftType>> &&
                        std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i=0; i < Size; ++i)
                A[i] = cgt::sbo(A[i]) / cgt::sbo(B); 

            return std::move(A);
        }
        else
        {
            c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A[i]) / cgt::sbo(B);

            return C; 
        }
    }

    template<cgt::numerical_c LeftType,  cgt::vector_c RightType>
        requires cgt::common_vector_c<LeftType, RightType>
    constexpr decltype(auto) operator/(LeftType A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using c_t = cgt::common_vector_t<LeftType, RightType>;

        std::size_t Size = B.size();

       if constexpr(std::same_as<c_t, std::remove_cvref_t<RightType>> &&
                        std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i=0; i < Size; ++i)
                B[i] = cgt::sbo(A) / cgt::sbo(B[i]); 
            
            return std::move(B);
        }
        else
        {
            c_t C(Size);

            for(std::size_t i=0; i < Size; ++i)
                C[i] = cgt::sbo(A) / cgt::sbo(B[i]);

            return C; 
        }
    }
}
// end of namespace std

#endif
// end of file 
