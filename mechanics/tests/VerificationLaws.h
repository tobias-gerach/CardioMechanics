#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "CBConstitutiveModelGuccione.h"
#include "CBConstitutiveModelHolzapfel.h"
#include "CBConstitutiveModelNeoHooke.h"

// The settings of the verification tests (tests/helpers/materials.py), with bulk modulus 10.
inline std::unique_ptr<CBConstitutiveModel> MakeLaw(const std::string &name, ParameterMap &parameters) {
    const std::string prefix = "Materials.Mat_1." + name + ".";
    std::unique_ptr<CBConstitutiveModel> law;
    if (name == "NeoHooke") {
        parameters.Set(prefix + "a", 1.0);
        parameters.Set(prefix + "k", 10.0);
        law = std::make_unique<CBConstitutiveModelNeoHooke>();
    } else if (name == "Holzapfel") {
        for (const auto &[key, value] : {std::pair<std::string, double>{"a", 1}, {"b", 1}, {"af", 1}, {"bf", 1}, {"as", 0.5},
                                         {"bs", 1}, {"afs", 0.3}, {"bfs", 1}, {"k", 10}, {"kappa", 10}})
            parameters.Set(prefix + key, value);
        law = std::make_unique<CBConstitutiveModelHolzapfel>();
    } else if (name == "Guccione") {
        for (const auto &[key, value] : {std::pair<std::string, double>{"C", 1}, {"bf", 8}, {"bt", 2}, {"bfs", 4}, {"K", 10}})
            parameters.Set(prefix + key, value);
        law = std::make_unique<CBConstitutiveModelGuccione>();
    } else {
        throw std::invalid_argument("no settings for material law " + name);
    }
    law->Init(&parameters, 1);
    return law;
}
