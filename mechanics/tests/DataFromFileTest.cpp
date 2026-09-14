#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "CBDataFromFile.h"

namespace {

// Writes one binary data set per time, the one of time k holding 10 k + i at index i, and the file
// list naming them, in the formats CBDataFromFile reads. Returns the path of the file list.
std::string WriteFileList(const std::string &name, const std::vector<TFloat> &times) {
    const std::string dir = testing::TempDir();
    std::ofstream list(dir + name + ".list");
    for (size_t k = 0; k < times.size(); ++k) {
        const std::string path = dir + name + "." + std::to_string(k) + ".dat";
        const int32_t count    = 2;
        const double values[2] = {10.0 * k, 10.0 * k + 1};
        std::ofstream data(path, std::ios::binary);
        data.write(reinterpret_cast<const char *>(&count), sizeof(count));
        data.write(reinterpret_cast<const char *>(values), sizeof(values));
        list << times[k] << " " << path << "\n";
    }
    return dir + name + ".list";
}

}  // namespace

// A solver that steps straight from an earlier interval onto the last listed time must get the
// last data set, as it does when an intermediate time has loaded the last interval first.
TEST(DataFromFile, LastListedTimeReachedFromAnEarlierInterval) {
    CBDataFromFile data;
    data.Init(WriteFileList("last", {0, 0.25, 0.5}));
    EXPECT_DOUBLE_EQ(data.Get(0.25, 0), 10);
    EXPECT_DOUBLE_EQ(data.Get(0.5, 0), 20);
    EXPECT_DOUBLE_EQ(data.Get(0.5, 1), 21);
    EXPECT_DOUBLE_EQ(data.Get(0.375, 0), 15);
}

// Past the last listed time there is no data, which the points-control plugin reads as no target. A
// later query within the list, as after the solver steps back, must get the listed data again.
TEST(DataFromFile, ListedDataReturnsAfterAQueryPastTheLastTime) {
    CBDataFromFile data;
    data.Init(WriteFileList("past", {0, 0.25, 0.5}));
    EXPECT_DOUBLE_EQ(data.Get(0.375, 0), 15);
    EXPECT_DOUBLE_EQ(data.Get(0.75, 0), 0);
    EXPECT_DOUBLE_EQ(data.Get(0.375, 0), 15);
}
