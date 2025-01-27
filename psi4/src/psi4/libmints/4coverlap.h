#pragma once

#include "psi4/libmints/integral.h"

namespace libint2 {
class Engine;
class Shell;
}  // namespace libint2

namespace psi {

class BasisSet;

class FourCenterOverlapInt {
   protected:
    std::shared_ptr<BasisSet> bs1_;
    std::shared_ptr<BasisSet> bs2_;
    std::shared_ptr<BasisSet> bs3_;
    std::shared_ptr<BasisSet> bs4_;

    std::unique_ptr<libint2::Engine> engine0_;

    std::vector<const double*> buffers_;
    std::vector<double> zero_vec_;

    void compute_quartet(const libint2::Shell& s1, const libint2::Shell& s2, const libint2::Shell& s3,
                         const libint2::Shell& s4);

   public:
    FourCenterOverlapInt(std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2, std::shared_ptr<BasisSet> bs3,
                         std::shared_ptr<BasisSet> bs4);
    ~FourCenterOverlapInt();

    std::shared_ptr<BasisSet> basis1();
    std::shared_ptr<BasisSet> basis2();
    std::shared_ptr<BasisSet> basis3();
    std::shared_ptr<BasisSet> basis4();

    virtual void compute_shell(int, int, int, int);
    const std::vector<const double*>& buffers() const { return buffers_; }
};

}  // namespace psi
