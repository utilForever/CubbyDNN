#ifndef CUBBYDNN_DOWNLOADER_HPP
#define CUBBYDNN_DOWNLOADER_HPP

#include <ostream>

namespace CubbyDNN
{
class Downloader final
{
 public:
    static bool DownloadData(const std::string& url, std::ostream& stream);

    static bool UnGzip(const std::string& gzFilename,
                       const std::string& outFilename);

 private:
    static std::size_t WriteCallback(void* ptr, std::size_t size,
                                     std::size_t nmemb, void* stream);
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_DOWNLOADER_HPP
