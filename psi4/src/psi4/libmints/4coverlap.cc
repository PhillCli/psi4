#include "4coverlap.h"
#include "psi4/libmints/basisset.h"
#include <libint2.hpp>

using namespace psi;

FourCenterOverlapInt::FourCenterOverlapInt(std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2,
                                           std::shared_ptr<BasisSet> bs3, std::shared_ptr<BasisSet> bs4)
    : bs1_(bs1), bs2_(bs2), bs3_(bs3), bs4_(bs4) {
    int maxam1 = bs1_->max_am();
    int maxam2 = bs2_->max_am();
    int maxam3 = bs3_->max_am();
    int maxam4 = bs4_->max_am();
    int max_am = std::max({maxam1, maxam2, maxam3, maxam4});
    int max_nprim = std::max({basis1()->max_nprimitive(), basis2()->max_nprimitive(), basis3()->max_nprimitive(),
                              basis4()->max_nprimitive()});

    int max_nao = INT_NCART(maxam1) * INT_NCART(maxam2) * INT_NCART(maxam3) * INT_NCART(maxam4);
    zero_vec_ = std::vector<double>(max_nao, 0.0);

    engine0_ = std::make_unique<libint2::Engine>(libint2::Operator::delta, max_nprim, max_am, 0, 0.0);
    buffers_.resize(1);
}

FourCenterOverlapInt::~FourCenterOverlapInt() {}

std::shared_ptr<BasisSet> FourCenterOverlapInt::basis1() { return bs1_; }
std::shared_ptr<BasisSet> FourCenterOverlapInt::basis2() { return bs2_; }
std::shared_ptr<BasisSet> FourCenterOverlapInt::basis3() { return bs3_; }
std::shared_ptr<BasisSet> FourCenterOverlapInt::basis4() { return bs4_; }

void FourCenterOverlapInt::compute_shell(int sh1, int sh2, int sh3, int sh4) {
    compute_quartet(bs1_->l2_shell(sh1), bs2_->l2_shell(sh2), bs3_->l2_shell(sh3), bs4_->l2_shell(sh4));
}

void FourCenterOverlapInt::compute_quartet(const libint2::Shell& s1, const libint2::Shell& s2, const libint2::Shell& s3,
                                           const libint2::Shell& s4) {
    engine0_->compute(s1, s2, s3, s4);
    buffers_[0] = engine0_->results()[0];
    if (buffers_[0] == nullptr) {
        buffers_[0] = zero_vec_.data();
    }
}
