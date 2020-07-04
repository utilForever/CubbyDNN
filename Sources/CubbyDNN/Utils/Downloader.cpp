#include <CubbyDNN/Utils/Downloader.hpp>

#include <curl/curl.h>
#include <zlib.h>
#include <fstream>
#include <sstream>

namespace CubbyDNN
{
bool Downloader::DownloadData(const std::string& url, std::ostream& stream)
{
    static bool curlInitialized = false;
    static CURL* curlHandle = nullptr;

    if (!curlInitialized)
    {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curlHandle = curl_easy_init();
        curlInitialized = true;

        curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION,
                         &Downloader::WriteCallback);
        curl_easy_setopt(curlHandle, CURLOPT_HEADER, 0L);
        curl_easy_setopt(curlHandle, CURLOPT_VERBOSE, 0L);
        curl_easy_setopt(curlHandle, CURLOPT_ACCEPT_ENCODING, "gzip");
        curl_easy_setopt(curlHandle, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curlHandle, CURLOPT_DNS_CACHE_TIMEOUT, -1);
    }

    curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, &stream);
    curl_easy_setopt(curlHandle, CURLOPT_URL, url.c_str());

    CURLcode result = curl_easy_perform(curlHandle);

    return result == CURLE_OK;
}

bool Downloader::UnGzip(const std::string& gzFilename,
                        const std::string& outFilename)
{
    gzFile fp = gzopen(gzFilename.c_str(), "rb");
    if (fp == nullptr)
        return false;

    std::ofstream ofs(outFilename, std::ios_base::binary);

    constexpr int BUF_SIZE = 1024;
    char buffer[BUF_SIZE];
    int readSize = 0;

    do
    {
        readSize = gzread(fp, buffer, BUF_SIZE);
        ofs.write(buffer, readSize);
    } while (!gzeof(fp));

    ofs.close();
    gzclose(fp);

    return true;
}

std::size_t Downloader::WriteCallback(void* ptr, std::size_t size,
                                      std::size_t nmemb, void* stream)
{
    const std::size_t count = size * nmemb;

    static_cast<std::ostringstream*>(stream)->write(
        reinterpret_cast<char*>(ptr), count);

    return count;
}
}  // namespace CubbyDNN
