#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "CBElementFactory.h"
#include "CBElementSurface.h"

namespace {

std::unique_ptr<CBElementSurface> NewSurface(const std::string &type, unsigned int numNodes) {
    CBElementFactory factory;
    std::unique_ptr<CBElement> element(factory.New(type, numNodes));
    if (!dynamic_cast<CBElementSurface *>(element.get()))
        return nullptr;
    return std::unique_ptr<CBElementSurface>(static_cast<CBElementSurface *>(element.release()));
}

struct RoleCase {
    std::string type;
    SurfaceRole role;
};

class SurfaceRoleTest : public ::testing::TestWithParam<RoleCase> {};

// A role is what the settings declare a surface for, and it survives the cloning the solver does
// on its copy of the model.
TEST_P(SurfaceRoleTest, TypeNameDeclaresTheRoleOfATriangle) {
    auto surface = NewSurface(GetParam().type, 3);
    ASSERT_NE(surface, nullptr);
    EXPECT_EQ(surface->GetRole(), GetParam().role);
    EXPECT_EQ(surface->GetNumberOfNodesIndices(), 3u);
    std::unique_ptr<CBElement> clone(surface->Clone());
    EXPECT_EQ(dynamic_cast<CBElementSurface &>(*clone).GetRole(), GetParam().role);
}

INSTANTIATE_TEST_SUITE_P(Roles, SurfaceRoleTest, ::testing::Values(
    RoleCase{"CAVITY", SurfaceRole::Cavity},
    RoleCase{"CONTACT_ROBIN", SurfaceRole::Robin},
    RoleCase{"CONTACT_MASTER", SurfaceRole::ContactMaster},
    RoleCase{"CONTACT_SLAVE", SurfaceRole::ContactSlave}));

TEST(SurfaceRole, ShapeNamesCarryNoRole) {
    auto t3 = NewSurface("T3", 3);
    ASSERT_NE(t3, nullptr);
    EXPECT_EQ(t3->GetRole(), SurfaceRole::None);
    auto t6 = NewSurface("T6", 6);
    ASSERT_NE(t6, nullptr);
    EXPECT_EQ(t6->GetRole(), SurfaceRole::None);
    EXPECT_EQ(t6->GetNumberOfNodesIndices(), 6u);
}

class SixNodeRoleTest : public ::testing::TestWithParam<RoleCase> {};

// A cavity and a Robin boundary keep a six-node face whole, so their loads follow the curved face
// with the quadratic shape functions.
TEST_P(SixNodeRoleTest, SixNodeFaceIsSixNodeElement) {
    auto surface = NewSurface(GetParam().type, 6);
    ASSERT_NE(surface, nullptr);
    EXPECT_EQ(surface->GetRole(), GetParam().role);
    EXPECT_EQ(surface->GetNumberOfNodesIndices(), 6u);
    std::unique_ptr<CBElement> clone(surface->Clone());
    EXPECT_EQ(dynamic_cast<CBElementSurface &>(*clone).GetRole(), GetParam().role);
}

INSTANTIATE_TEST_SUITE_P(Roles, SixNodeRoleTest, ::testing::Values(
    RoleCase{"CAVITY", SurfaceRole::Cavity},
    RoleCase{"CONTACT_ROBIN", SurfaceRole::Robin}));

// Contact is carried by three-node triangles, into which the loader refines a six-node face.
TEST(SurfaceRole, ContactOnSixNodeFaceIsThreeNodeElement) {
    EXPECT_EQ(NewSurface("CONTACT_MASTER", 6)->GetNumberOfNodesIndices(), 3u);
    EXPECT_EQ(NewSurface("CONTACT_SLAVE", 6)->GetNumberOfNodesIndices(), 3u);
}

}  // namespace
