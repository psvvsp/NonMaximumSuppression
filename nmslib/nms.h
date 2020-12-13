#include "nms_interface.h"

class NMS_impl;

class NMS : public NMS_interface
{
public:
    NMS();
    virtual ~NMS();

    NMS(const NMS& other) = delete;
    NMS& operator =(const NMS& other) = delete;

    bool doIt(
        const std::vector<Box>& boxesIn,
        const std::vector<real>& scoresIn,
        real threshold,
        std::vector<Box>& boxesOut,
        std::vector<real>& scoresOut
    ) override;

    bool init(size_t boxesCountMax) override;

private:
    NMS_impl *m_impl;
};