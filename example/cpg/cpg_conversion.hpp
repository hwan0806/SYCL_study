/**
 * @file tpf_conversion.hpp
 * @author Thomas Kim (ThomasKim@TalkPlayFun.com)
 * @brief String conversions are implemented.
 * @version 0.1
 * @date 2019-04-13
 *
 * @copyright Copyright (c) 2019
 *
 */

#ifndef _CPG_CONVERSION_HPP
#define _CPG_CONVERSION_HPP

#ifndef TBB_SUPPRESS_DEPRECATED_MESSAGES
#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#endif // end of TBB_SUPPRESS_DEPRECATED_MESSAGES

#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifdef _MSVC_LANG
#if _MSVC_LANG < 201703L
#error This libary requires C++17 Standard (Visual Studio 2017).
#endif
#else

#if __cplusplus < 201703
#error This library requires C++17 Standard (GNU g++ version 8.0 or clang++ version 8.0 above)
#endif // end of __cplusplus

#endif // end of _MSVC_LANG

#include <clocale>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>


#ifdef _WIN32
#include <windows.h>
#else
// Linux/Unix 환경에서 Windows 상수들을 정의
#define CP_UTF8 65001
#define CP_ACP 0
#endif

/**
 * @brief Includes subnamespace conversion.
 *
 */
namespace cpg {
/**
 * @brief String conversions are implemented.
 *
 */
namespace conversion {
/**
 * @brief Load system default locale for string conversion.
 *
 * @sa <a target ="_blank"
 * href="002-conversion_8cpp_source.html">002-conversion.cpp</a>
 */
inline void load_default_locale(bool ShowLocaleName = false) {
#ifdef _WIN32
  // GetSystemDefaultLocaleName function :
  // https://goo.gl/WLLSG3

  // allocate buffer to hold locale name
  std::wstring locale_name(LOCALE_NAME_MAX_LENGTH, L'\0');

  int locale_legnth =
      GetSystemDefaultLocaleName(&locale_name[0], LOCALE_NAME_MAX_LENGTH);

  // if failed to get locale name, then just return
  if (locale_legnth == 0)
    return;

  // trim trailing buffer
  if (locale_name[locale_legnth] == L'\0')
    locale_name = locale_name.substr(0, locale_legnth - 1);

  // https://goo.gl/1A2Hh4
  // set locale of the process or thread
  _wsetlocale(LC_ALL, locale_name.c_str());

  if (ShowLocaleName)
    std::wcout << L"locale name: " << locale_name << std::endl;
#else
  // Linux/Unix 환경에서는 기본 로케일 설정
  setlocale(LC_ALL, "");
  if (ShowLocaleName)
    std::cout << "locale: " << setlocale(LC_ALL, nullptr) << std::endl;
#endif
}

/**
 * @brief Converts from std::wstring to std::string with codepage
 *
 * If fails, returns empty string, std::string("")
 *
 * @param wstr for std::wstring to convert
 * @param codepage for Code Page, default = CP_UTF8
 * @return std::string for the converted string
 *
 * @sa <a target ="_blank"
 * href="002-conversion_8cpp_source.html">002-conversion.cpp</a>
 *
 */
inline std::string wstring_to_string(const std::wstring &wstr,
                                     unsigned int codepage = CP_UTF8) {
  if (wstr.empty())
    return "";

#ifdef _WIN32
  // https://goo.gl/acoQBt
  int str_len =
      WideCharToMultiByte(codepage,
                          0, // dwFlag
                          wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);

  // if failed to compute the byte count of the string
  if (str_len == 0)
    return "";

  // prepare a string buffer to
  // hold the converted string
  // do not +1 to str_len,
  // because std::string manages terminating null
  std::string str(str_len, '\0');

  int converted =
      WideCharToMultiByte(codepage, 0, wstr.c_str(), (int)wstr.size(), &str[0],
                          str_len, NULL, NULL);

  return (converted == 0 ? "" : str);
#else
  // Linux/Unix 환경에서는 wcstombs 사용
  std::string str;
  str.reserve(wstr.size() * 4); // UTF-8은 최대 4바이트

  for (wchar_t wc : wstr) {
    char mb[MB_CUR_MAX];
    int len = wctomb(mb, wc);
    if (len > 0) {
      str.append(mb, len);
    }
  }
  return str;
#endif
}

/** @example 002-conversion.cpp
 *
 * @brief This is an example of how to use wstring_to_string().
 *
 */

/**
 * @brief Converts from std::string to std::wstring with codepage
 *
 * If fails, returns empty string, std::wstring(L"")
 *
 * @param str for std::string to convert
 * @param codepage for Code Page, default = CP_UTF8
 * @return std::wstring for the converted string
 *
 * @sa <a target ="_blank"
 * href="002-conversion_8cpp_source.html">002-conversion.cpp</a>
 *
 */
inline std::wstring string_to_wstring(const std::string &str,
                                      unsigned int codepage = CP_UTF8) {
  if (str.empty())
    return L"";

#ifdef _WIN32
  // https://goo.gl/acoQBt
  int wstr_len = MultiByteToWideChar(codepage,
                                     0, // dwFlag
                                     str.c_str(), (int)str.size(), NULL, 0);

  // if failed to compute the byte count of the string
  if (wstr_len == 0)
    return L"";

  // prepare a wstring buffer to
  // hold the converted string
  // do not +1 to wstr_len,
  // because std::wstring manages terminating null
  std::wstring wstr(wstr_len, L'\0');

  int converted = MultiByteToWideChar(codepage, 0, str.c_str(), (int)str.size(),
                                      &wstr[0], wstr_len);

  return (converted == 0 ? L"" : wstr);
#else
  // Linux/Unix 환경에서는 mbstowcs 사용
  std::wstring wstr;
  wstr.reserve(str.size());

  const char *mbstr = str.c_str();
  size_t len = mbstowcs(nullptr, mbstr, 0);
  if (len == static_cast<size_t>(-1))
    return L"";

  wstr.resize(len);
  mbstowcs(&wstr[0], mbstr, len);
  return wstr;
#endif
}

/**
 * @brief Converts from UTF-16 to UTF-8
 *
 * @param utf16str for UTF-16 string to convert
 * @return std::string for the converted UTF-8 string
 *
 */
inline std::string utf16_to_utf8(const std::wstring &utf16str) {
  return wstring_to_string(utf16str, CP_UTF8);
}

/**
 * @brief Converts from UTF-8 to UTF-16
 *
 * @param utf8str for UTF-8 string to convert
 * @return std::wstring for the converted UTF-16 string
 *
 */
inline std::wstring utf8_to_utf16(const std::string &utf8str) {
  return string_to_wstring(utf8str, CP_UTF8);
}

/**
 * @brief Converts from UTF-16 to UTF-8
 *
 * @param utf16str for UTF-16 string to convert
 * @return std::string for the converted UTF-8 string
 *
 */
inline std::string utf16_to_utf8(const wchar_t *utf16str) {
  if (utf16str == nullptr)
    return "";
  return wstring_to_string(std::wstring(utf16str), CP_UTF8);
}

/**
 * @brief Converts from UTF-8 to UTF-16
 *
 * @param utf8str for UTF-8 string to convert
 * @return std::wstring for the converted UTF-16 string
 *
 */
inline std::wstring utf8_to_utf16(const char *utf8str) {
  if (utf8str == nullptr)
    return L"";
  return string_to_wstring(std::string(utf8str), CP_UTF8);
}

/**
 * @brief Converts from UTF-16 character to UTF-8 character
 *
 * @param utf16_char for UTF-16 character to convert
 * @return char for the converted UTF-8 character
 *
 */
inline char utf16_to_utf8(const wchar_t utf16_char) {
  std::wstring wstr(1, utf16_char);
  std::string str = wstring_to_string(wstr, CP_UTF8);
  return str.empty() ? '\0' : str[0];
}

/**
 * @brief Converts from UTF-8 character to UTF-16 character
 *
 * @param utf8_char for UTF-8 character to convert
 * @return wchar_t for the converted UTF-16 character
 *
 */
inline wchar_t utf8_to_utf16(const char utf8_char) {
  std::string str(1, utf8_char);
  std::wstring wstr = string_to_wstring(str, CP_UTF8);
  return wstr.empty() ? L'\0' : wstr[0];
}

/**
 * @brief Converts from Windows codepage to UTF-16
 *
 * @param codepage_string for Windows codepage string to convert
 * @return std::wstring for the converted UTF-16 string
 *
 */
inline std::wstring
windows_codepage_to_utf16(const std::string &codepage_string) {
  return string_to_wstring(codepage_string, CP_ACP);
}

/**
 * @brief Converts from UTF-16 to Windows codepage
 *
 * @param utf16_string for UTF-16 string to convert
 * @return std::string for the converted Windows codepage string
 *
 */
inline std::string utf16_to_windows_codepage(const std::wstring &utf16_string) {
  return wstring_to_string(utf16_string, CP_ACP);
}

/**
 * @brief Converts from Windows codepage to UTF-16
 *
 * @param codepage_string for Windows codepage string to convert
 * @return std::wstring for the converted UTF-16 string
 *
 */
inline std::wstring windows_codepage_to_utf16(const char *codepage_string) {
  if (codepage_string == nullptr)
    return L"";
  return string_to_wstring(std::string(codepage_string), CP_ACP);
}

/**
 * @brief Converts from UTF-16 to Windows codepage
 *
 * @param utf16_string for UTF-16 string to convert
 * @return std::string for the converted Windows codepage string
 *
 */
inline std::string utf16_to_windows_codepage(const wchar_t *utf16_string) {
  if (utf16_string == nullptr)
    return "";
  return wstring_to_string(std::wstring(utf16_string), CP_ACP);
}

/**
 * @brief Converts from Windows codepage to UTF-8
 *
 * @param codepage_string for Windows codepage string to convert
 * @return std::string for the converted UTF-8 string
 *
 */
inline std::string
windows_codepage_to_utf8(const std::string &codepage_string) {
  std::wstring utf16 = windows_codepage_to_utf16(codepage_string);
  return utf16_to_utf8(utf16);
}

/**
 * @brief Converts from Windows codepage to UTF-8
 *
 * @param codepage_string for Windows codepage string to convert
 * @return std::string for the converted UTF-8 string
 *
 */
inline std::string windows_codepage_to_utf8(const char *codepage_string) {
  if (codepage_string == nullptr)
    return "";
  std::wstring utf16 = windows_codepage_to_utf16(codepage_string);
  return utf16_to_utf8(utf16);
}

/**
 * @brief Converts from UTF-8 to Windows codepage
 *
 * @param utf8_string for UTF-8 string to convert
 * @return std::string for the converted Windows codepage string
 *
 */
inline std::string utf8_to_windows_codepage(const std::string &utf8_string) {
  std::wstring utf16 = utf8_to_utf16(utf8_string);
  return utf16_to_windows_codepage(utf16);
}

/**
 * @brief Converts from UTF-8 to Windows codepage
 *
 * @param utf8_string for UTF-8 string to convert
 * @return std::string for the converted Windows codepage string
 *
 */
inline std::string utf8_to_windows_codepage(const char *utf8_string) {
  if (utf8_string == nullptr)
    return "";
  std::wstring utf16 = utf8_to_utf16(utf8_string);
  return utf16_to_windows_codepage(utf16);
}

/**
 * @brief Template function for source to target conversion
 *
 * @param str for source string to convert
 * @return decltype(auto) for the converted target string
 *
 */
template <typename TargetCharType, typename SourceCharType>
decltype(auto) source_to_target(const std::basic_string<SourceCharType> &str) {
  if constexpr (std::is_same_v<TargetCharType, char> &&
                std::is_same_v<SourceCharType, wchar_t>)
    return wstring_to_string(str);
  else if constexpr (std::is_same_v<TargetCharType, wchar_t> &&
                     std::is_same_v<SourceCharType, char>)
    return string_to_wstring(str);
  else
    return str;
}

/**
 * @brief Smart encode function for char*
 *
 * @param arg for char* to encode
 * @return std::string for the encoded string
 *
 */
inline std::string smart_encode(const char *arg) {
  if (arg == nullptr)
    return "";
  return std::string(arg);
}

/**
 * @brief Smart encode function for wchar_t*
 *
 * @param arg for wchar_t* to encode
 * @return std::string for the encoded string
 *
 */
inline std::string smart_encode(const wchar_t *arg) {
  if (arg == nullptr)
    return "";
  return wstring_to_string(std::wstring(arg));
}
} // namespace conversion
} // namespace cpg

#endif // end of file _CPG_CONVERSION_HPP