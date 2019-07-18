//
// Created by jwkim98 on 7/9/19.
//

#ifndef CUBBYDNN_RECORDER_HPP
#define CUBBYDNN_RECORDER_HPP

#include <deque>
#include <string>
#include <vector>

namespace CubbyDNN
{
enum class MessageType
{
    Error,
    Warning,
    Output,
};

struct MessageUnit
{
    size_t ID;
    size_t Name;
    std::deque<std::string> MsgQueue;

    void RecordMessage(MessageType msgType, const std::string& message)
    {
        std::string msg;
        switch (msgType)
        {
            case MessageType::Error:
                msg += "Error : ";
                break;
            case MessageType::Warning:
                msg += "Warning : ";
                break;
            case MessageType::Output:
                msg += "Output : ";
                break;
        }
        MsgQueue.emplace_back(message);
    }
};

}  // namespace CubbyDNN

#endif  //CUBBYDNN_RECORDER_HPP
