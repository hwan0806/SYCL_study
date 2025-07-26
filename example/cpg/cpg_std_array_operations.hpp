/*
    Author: Thomas Kim
    First Edit: Dec. 03, 2021
    Second Edit: Dec. 07, 2021
    Third Edit: Dec. 09, 2021 - safe binary operation
*/

#ifndef _CGP_STD_ARRAY_OPERATIONS_HPP
#define _CGP_STD_ARRAY_OPERATIONS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_types.hpp"

namespace std::inline stl_extensions
{
    namespace cgt = cpg::types;
   
    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 
            std::tuple_size_v<std::remove_cvref_t<RightType>>) && 
            (cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>) ||
            (cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>) &&
            cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator+(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = cgt::sbo(A[i]) + cgt::sbo(B[i]);
            
            return std::move(A);
        }
        else if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = cgt::sbo(A[i]) + cgt::sbo(B[i]);
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(A[i]) + cgt::sbo(B[i]);

            return C;
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 
            std::tuple_size_v<std::remove_cvref_t<RightType>>) && 
            (cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>) ||
            (cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>) &&
            cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator-(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = cgt::sbo(A[i]) - cgt::sbo(B[i]);
            
            return std::move(A);
        }
        else if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = cgt::sbo(A[i]) - cgt::sbo(B[i]);
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(A[i]) - cgt::sbo(B[i]);

            return C;
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 
            std::tuple_size_v<std::remove_cvref_t<RightType>>) && 
            (cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>) ||
            (cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>) &&
            cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator*(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = cgt::sbo(A[i]) * cgt::sbo(B[i]);
            
            return std::move(A);
        }
        else if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = cgt::sbo(A[i]) * cgt::sbo(B[i]);
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(A[i]) * cgt::sbo(B[i]);

            return C;
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 
            std::tuple_size_v<std::remove_cvref_t<RightType>>) && 
            (cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>) ||
            (cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>) &&
            cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator/(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = cgt::sbo(A[i]) / cgt::sbo(B[i]);
            
            return std::move(A);
        }
        else if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = cgt::sbo(A[i]) / cgt::sbo(B[i]);
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(A[i]) / cgt::sbo(B[i]);

            return C;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // vector cross product - operator% has high precedence
    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 
            std::tuple_size_v<std::remove_cvref_t<RightType>>) &&
            ( (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 3) || (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 4))
                && cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>>
                && cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>
                && cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator%(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

        C[0] = cgt::sbo(A[1]) * cgt::sbo(B[2]) - cgt::sbo(B[1]) * cgt::sbo(A[2]);
        C[1] = cgt::sbo(A[2]) * cgt::sbo(B[0]) - cgt::sbo(B[2]) * cgt::sbo(A[0]);
        C[2] = cgt::sbo(A[0]) * cgt::sbo(B[1]) - cgt::sbo(B[0]) * cgt::sbo(A[1]);

        if constexpr(std::tuple_size_v<std::remove_cvref_t<LeftType>> == 4)
            C[3] = 1;

        return C;
    }
     
    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 
            std::tuple_size_v<std::remove_cvref_t<RightType>>) && 
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&  
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator%(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = A[i] % B[i];
            
            return std::move(A);
        }
        else if constexpr(std::same_as<container_t_a, container_t_b> && 
            std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = A[i] % B[i];
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A[i] % B[i];

            return C;
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&  
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator%(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = A[i] % B;
            
            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A[i] % B;

            return C;
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&  
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator%(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<RightType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = A % B[i];
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            // std::cout <<"A    type: " >> A << std::endl;
            // std::cout <<"B[0] type: " >> B[0] << std::endl;
            // std::cout <<"C[0] type: " >> C[0] << std::endl;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A % B[i];

            return C;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // vector inner product - operator& has low precedence
    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 
            std::tuple_size_v<std::remove_cvref_t<RightType>>)
            && cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>>
            && cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>
            && cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator&(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        using c_t = cgt::common_std_array_t<LeftType, RightType>;

        using element_t = std::tuple_element_t<0, c_t>;

        element_t R{};

        for(std::size_t i=0; i < N; ++i)
            R += cgt::sbo(A[i]) * cgt::sbo(B[i]);

        return R;
    }
    
    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires (std::tuple_size_v<std::remove_cvref_t<LeftType>> == 
            std::tuple_size_v<std::remove_cvref_t<RightType>>) && 
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&  
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator&(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        using c_t = cgt::common_std_array_t<LeftType, RightType>;
        using element_t = std::tuple_element_t<0, std::tuple_element_t<0, c_t> >;
        using array_t = std::array<element_t, N>;

        array_t C;

        for(std::size_t i=0; i < N; ++i)
            C[i] = A[i] & B[i];

        return C;
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&  
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>
    constexpr decltype(auto) operator&(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using a_t = std::tuple_element_t<0, std::tuple_element_t<0, container_t_a>>;

        using container_t_b = std::remove_cvref_t<RightType>;
        using b_t = std::tuple_element_t<0, container_t_b>;

        constexpr std::size_t N = std::tuple_size_v<container_t_a>;
        
        using c_t = std::array<cgt::common_signed_t<a_t, b_t>, N>; c_t C;

        for(std::size_t i=0; i < N; ++i) C[i] = A[i] & B;

        return C;
    }

   template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&  
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>>
    constexpr decltype(auto) operator&(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using a_t = std::tuple_element_t<0, container_t_a>;

        using container_t_b = std::remove_cvref_t<RightType>;
        using b_t = std::tuple_element_t<0, std::tuple_element_t<0, container_t_b>>;

        constexpr std::size_t N = std::tuple_size_v<container_t_b>;
        
        using c_t = std::array<cgt::common_signed_t<a_t, b_t>, N>; c_t C;

        // std::cout <<"A    type: " >> A << std::endl;
        // std::cout <<"B[0] type: " >> B[0] << std::endl;
        // std::cout <<"C[0] type: " >> C[0] << std::endl;

        for(std::size_t i=0; i < N; ++i) C[i] = A & B[i];

        return C;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
     template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator+(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = A[i] + B;
            
            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A[i] + B;

            return C;
        }
    }

     template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator+(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<RightType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = A + B[i];
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A + B[i];

            return C;
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
     template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator-(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = A[i] - B;
            
            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A[i] - B;

            return C;
        }
    }

     template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator-(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<RightType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = A - B[i];
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A - B[i];

            return C;
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator*(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = A[i] * B;
            
            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A[i] * B;

            return C;
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator*(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<RightType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               B[i] = A * B[i];
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A * B[i];

            return C;
        }
    }

    template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator/(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<LeftType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
               A[i] = A[i] / B;
            
            return std::move(A);
        }
        else
        {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = A[i] / B;

            return C;
        }
    }

     template<cgt::std_array_flat_c LeftType, cgt::std_array_flat_c RightType>
        requires cgt::numerical_c<std::tuple_element_t<0, std::remove_cvref_t<LeftType>>> &&
            cgt::std_array_c<std::tuple_element_t<0, std::remove_cvref_t<RightType>>> &&
                cgt::common_std_array_c<LeftType, RightType>
    constexpr decltype(auto) operator/(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t_a = std::remove_cvref_t<LeftType>;
        using container_t_b = std::remove_cvref_t<RightType>;
        
        constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<RightType>>;
        
        if constexpr(std::is_rvalue_reference_v<decltype(B)>)
        {
            for(std::size_t i{}; i < N; ++i) B[i] = A / B[i];
            
            return std::move(B);
        }
        else
       {
            using c_t = cgt::common_std_array_t<LeftType, RightType>; c_t C;

            for(std::size_t i=0; i < N; ++i) C[i] = A / B[i];

            return C;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    template<cgt::std_array_flat_c ArrayType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator+(ArrayType&& A, ScalarType s) noexcept(!cpg::bDetectOverFlow)
    {
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<ArrayType>>;

        using c_t = cgt::common_std_array_t<ArrayType, ScalarType>;

        if constexpr(std::same_as<c_t, std::remove_cvref_t<ArrayType>> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
                A[i] = cgt::sbo(A[i]) + cgt::sbo(s);

            return std::move(A);
        }
        else
        {
            c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(A[i]) + cgt::sbo(s);     
            
            return C; 
        }
    }

    template<cgt::numerical_c ScalarType, cgt::std_array_flat_c ArrayType>
    constexpr decltype(auto) operator+(ScalarType s, ArrayType&& A) noexcept(!cpg::bDetectOverFlow)
    {
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<ArrayType>>;

        using c_t = cgt::common_std_array_t<ScalarType, ArrayType>;

        if constexpr(std::same_as<c_t, std::remove_cvref_t<ArrayType>> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
                A[i] = cgt::sbo(s) + cgt::sbo(A[i]);

            return std::move(A);
        }
        else
        {
            c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(s) + cgt::sbo(A[i]);
            
            return C; 
        }
    }

    template<cgt::std_array_flat_c ArrayType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator-(ArrayType&& A, ScalarType s) noexcept(!cpg::bDetectOverFlow)
    {
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<ArrayType>>;

        using c_t = cgt::common_std_array_t<ArrayType, ScalarType>;

        if constexpr(std::same_as<c_t, std::remove_cvref_t<ArrayType>> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
                A[i] = cgt::sbo(A[i]) - cgt::sbo(s);

            return std::move(A);
        }
        else
        {
            c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(A[i]) - cgt::sbo(s);     
            
            return C; 
        }
    }

    template<cgt::numerical_c ScalarType, cgt::std_array_flat_c ArrayType>
    constexpr decltype(auto) operator-(ScalarType s, ArrayType&& A) noexcept(!cpg::bDetectOverFlow)
    {
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<ArrayType>>;

        using c_t = cgt::common_std_array_t<ScalarType, ArrayType>;

        if constexpr(std::same_as<c_t, std::remove_cvref_t<ArrayType>> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
                A[i] = cgt::sbo(s) - cgt::sbo(A[i]);

            return std::move(A);
        }
        else
        {
            c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(s) - cgt::sbo(A[i]);
            
            return C; 
        }
    }

    template<cgt::std_array_flat_c ArrayType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator*(ArrayType&& A, ScalarType s) noexcept(!cpg::bDetectOverFlow)
    {
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<ArrayType>>;

        using c_t = cgt::common_std_array_t<ArrayType, ScalarType>;

        if constexpr(std::same_as<c_t, std::remove_cvref_t<ArrayType>> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
                A[i] = cgt::sbo(A[i]) * cgt::sbo(s);

            return std::move(A);
        }
        else
        {
            c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(A[i]) * cgt::sbo(s);     
            
            return C; 
        }
    }

    template<cgt::numerical_c ScalarType, cgt::std_array_flat_c ArrayType>
    constexpr decltype(auto) operator*(ScalarType s, ArrayType&& A) noexcept(!cpg::bDetectOverFlow)
    {
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<ArrayType>>;

        using c_t = cgt::common_std_array_t<ScalarType, ArrayType>;

        if constexpr(std::same_as<c_t, std::remove_cvref_t<ArrayType>> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
                A[i] = cgt::sbo(s) * cgt::sbo(A[i]);

            return std::move(A);
        }
        else
        {
            c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(s) * cgt::sbo(A[i]);
            
            return C; 
        }
    }

    template<cgt::std_array_flat_c ArrayType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator/(ArrayType&& A, ScalarType s) noexcept(!cpg::bDetectOverFlow)
    {
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<ArrayType>>;

        using c_t = cgt::common_std_array_t<ArrayType, ScalarType>;

        if constexpr(std::same_as<c_t, std::remove_cvref_t<ArrayType>> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
                A[i] = cgt::sbo(A[i]) / cgt::sbo(s);

            return std::move(A);
        }
        else
        {
            c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(A[i]) / cgt::sbo(s);     
            
            return C; 
        }
    }

    template<cgt::numerical_c ScalarType, cgt::std_array_flat_c ArrayType>
    constexpr decltype(auto) operator/(ScalarType s, ArrayType&& A) noexcept(!cpg::bDetectOverFlow)
    {
        constexpr std::size_t N = 
            std::tuple_size_v<std::remove_cvref_t<ArrayType>>;

        using c_t = cgt::common_std_array_t<ScalarType, ArrayType>;

        if constexpr(std::same_as<c_t, std::remove_cvref_t<ArrayType>> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            for(std::size_t i{}; i < N; ++i) 
                A[i] = cgt::sbo(s) / cgt::sbo(A[i]);

            return std::move(A);
        }
        else
        {
            c_t C;

            for(std::size_t i=0; i < N; ++i)
                C[i] = cgt::sbo(s) / cgt::sbo(A[i]);
            
            return C; 
        }
    }
}

#endif
// end of file 
