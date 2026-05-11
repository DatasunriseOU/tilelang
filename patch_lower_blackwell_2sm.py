import sys

with open("src/transform/lower_blackwell_2sm.cc", "r") as f:
    code = f.read()

import re

old_code = """class Tcgen5_2SmLower : public StmtExprMutator {
public:
  Tcgen5_2SmLower(bool cluster_dims_valid)
      : cluster_dims_valid_(cluster_dims_valid) {}
  bool has_2sm_tcgen5mma() const { return has_2sm_tcgen5mma_; }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      TileOperator tile_op = ParseOperator(ffi::GetRef<Stmt>(op));
      if (tile_op.defined() && tile_op.as<Gemm>()) {
        // Check if the user explicitly requested 2CTA via the use_2cta
        // annotation on the Call node (set by T.tcgen05_gemm(use_2cta=True)).
        if (call->annotations.count(attr::kUse2Cta)) {
          auto val = call->annotations.Get(attr::kUse2Cta).value();
          if (const auto *imm = val.as<IntImmNode>()) {
            if (imm->value) {
              if (!cluster_dims_valid_) {
                LOG(WARNING) << "Invalid cluster_dims disables 2CTA "
                                "TCGEN5MMA, use 1CTA variant instead.";
                return StmtExprMutator::VisitStmt_(op);
              }
              has_2sm_tcgen5mma_ = true;
            }
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  bool cluster_dims_valid_;
  bool has_2sm_tcgen5mma_ = false;
};

class Tcgen5_2SmAnnotator : public StmtExprMutator {
public:
  explicit Tcgen5_2SmAnnotator() {}

private:
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    Stmt new_realize = StmtExprMutator::VisitStmt_(op);
    if (root_block_annotated_)
      return new_realize;
    const auto *realize = new_realize.as<SBlockRealizeNode>();
    ICHECK(realize);
    SBlock block = realize->block;
    SBlockNode *n = block.CopyOnWrite();
    // Set block attr: {use_2cta: 1}
    // lower_shared_tmem.cc will depend on this to allocate/deallocate tmem with
    // 2cta.
    n->annotations.Set(attr::kUse2Cta, IntImm(DataType::Int(32), 1));
    root_block_annotated_ = true;
    return SBlockRealize(realize->iter_values, realize->predicate, block);
  }

  bool root_block_annotated_ = false;
};

using namespace tirx::transform;

tvm::transform::Pass LowerBlackwell2SM() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetIsSm100(opt_target.value())) {
      return f;
    }
    Stmt body = f->body;
    bool cluster_dims_valid = HasValidClusterDimsFor2Cta(body);
    Tcgen5_2SmLower lower(cluster_dims_valid);
    body = lower(std::move(body));
    if (lower.has_2sm_tcgen5mma()) {
      // Annotate block attr for using 2cta tcgen5
      Tcgen5_2SmAnnotator annotator;
      body = annotator(std::move(body));
    }
    return PrimFunc(f->params, body, f->ret_type, f->buffer_map, f->attrs);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerBlackwell2SM", {});
}"""

new_code = """class Tcgen5_2SmAnnotator : public StmtExprMutator {
public:
  Tcgen5_2SmAnnotator(bool cluster_dims_valid)
      : cluster_dims_valid_(cluster_dims_valid) {}

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      TileOperator tile_op = ParseOperator(ffi::GetRef<Stmt>(op));
      if (tile_op.defined() && tile_op.as<Gemm>()) {
        // Check if the user explicitly requested 2CTA via the use_2cta
        // annotation on the Call node (set by T.tcgen05_gemm(use_2cta=True)).
        if (call->annotations.count(attr::kUse2Cta)) {
          auto val = call->annotations.Get(attr::kUse2Cta).value();
          if (const auto *imm = val.as<IntImmNode>()) {
            if (imm->value) {
              if (!cluster_dims_valid_) {
                LOG(WARNING) << "Invalid cluster_dims disables 2CTA "
                                "TCGEN5MMA, use 1CTA variant instead.";
                return StmtExprMutator::VisitStmt_(op);
              }
              has_2sm_tcgen5mma_in_current_block_ = true;
            }
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    bool old_has = has_2sm_tcgen5mma_in_current_block_;
    has_2sm_tcgen5mma_in_current_block_ = false;

    Stmt new_realize = StmtExprMutator::VisitStmt_(op);

    bool contains_2cta = has_2sm_tcgen5mma_in_current_block_;
    // Restore the state for the outer block
    has_2sm_tcgen5mma_in_current_block_ = old_has;

    if (contains_2cta) {
      const auto *realize = new_realize.as<SBlockRealizeNode>();
      ICHECK(realize);
      SBlock block = realize->block;
      SBlockNode *n = block.CopyOnWrite();
      // Set block attr: {use_2cta: 1}
      // lower_shared_tmem.cc will depend on this to allocate/deallocate tmem with
      // 2cta.
      n->annotations.Set(attr::kUse2Cta, IntImm(DataType::Int(32), 1));
      return SBlockRealize(realize->iter_values, realize->predicate, block);
    }
    return new_realize;
  }

  bool cluster_dims_valid_;
  bool has_2sm_tcgen5mma_in_current_block_ = false;
};

using namespace tirx::transform;

tvm::transform::Pass LowerBlackwell2SM() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetIsSm100(opt_target.value())) {
      return f;
    }
    Stmt body = f->body;
    bool cluster_dims_valid = HasValidClusterDimsFor2Cta(body);
    Tcgen5_2SmAnnotator annotator(cluster_dims_valid);
    body = annotator(std::move(body));
    return PrimFunc(f->params, body, f->ret_type, f->buffer_map, f->attrs);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerBlackwell2SM", {});
}"""

if old_code in code:
    with open("src/transform/lower_blackwell_2sm.cc", "w") as f:
        f.write(code.replace(old_code, new_code))
    print("Patch applied successfully.")
else:
    print("Old code not found. Check exact formatting.")

