"""MainWindow's spec <-> widgets copy (SpecsMixin, moved from app.py): _load_into_widgets puts the five jobs'
specs (Data, Grids, Pipeline, Custom snapshot, Plots) into the tabs' widgets, _read_widgets reads the
widgets back into them, and _set_opt / _get_opt do the same for one option of a figure set."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

import runspec as rs
from app_widgets import _checked, _int


class SpecsMixin:
    @staticmethod
    def _set_opt(o: rs.Opt, w: QWidget, value):
        if o.kind == "bool":
            w.setChecked(bool(value))
        elif o.kind == "int":
            w.setValue(int(value))
        elif o.kind == "float":
            w.setValue(float(value))
        elif o.kind == "choice":
            w.setCurrentText(str(value))
        elif o.kind == "plane":
            w.setCurrentIndex(max(w.findData(str(value)), 0))
        elif o.kind == "multi":
            want = {v.strip() for v in str(value).split(",") if v.strip()}
            for i in range(w.count()):
                w.item(i).setCheckState(Qt.Checked if w.item(i).text() in want else Qt.Unchecked)
        else:
            w.setText(str(value))

    @staticmethod
    def _get_opt(o: rs.Opt, w: QWidget):
        if o.kind == "bool":
            return w.isChecked()
        if o.kind in ("int", "float"):
            return w.value()
        if o.kind == "choice":
            return w.currentText()
        if o.kind == "plane":
            return w.currentData() or ""
        if o.kind == "multi":
            return ",".join(w.item(i).text() for i in range(w.count()) if w.item(i).checkState() == Qt.Checked)
        return w.text().strip()

    # ================================================================ spec <-> widgets
    def _load_into_widgets(self):
        self._loading = True
        s, p, pl = self.spec, self.pipe, self.plots
        self.root_edit.setText(s.data_root)
        d = self.data
        d.sim = d.sim or s.sim
        for cb, on in ((self.d_dl_snap, d.download_snapshots), (self.d_dl_gc, d.download_groupcats),
                       (self.d_dl_tree, d.download_trees), (self.d_dl_ics, d.download_ics),
                       (self.d_mg_snap, d.merge_snapshots), (self.d_mg_gc, d.merge_groupcats),
                       (self.d_mg_tree, d.merge_trees), (self.d_mg_ics, d.convert_ics),
                       (self.d_delete, d.delete_chunks)):
            cb.setChecked(on)
        self._fill_sims()
        # Grids
        (self.ps_radio if s.estimator == "ps" else self.dtfe_radio).setChecked(True)
        self.deposit_combo.setCurrentIndex(max(self.deposit_combo.findData(s.deposit), 0))
        self.grid_combo.setCurrentText(str(s.grid))
        self.nsub_spin.setValue(s.nsub)
        self.gpu_check.setChecked(s.gpu)
        for name, cb in self.field_checks.items():
            cb.setChecked(name in s.fields)
        self.vm_check.setChecked(s.vertex_mass)
        self.vw_check.setChecked(s.volume_weighted)
        self.ca_check.setChecked(s.caustics)
        self.cu_check.setChecked(s.caustic_cusps)
        self.ld_check.setChecked(s.linear_deposit)
        self.vg_check.setChecked(s.slice_vel_grad)
        self.pc_combo.setCurrentIndex(max(self.pc_combo.findData(s.pts_conventions), 0))
        self.part_spin.setValue(s.partition)
        self.conc_spin.setValue(s.max_concurrent)
        self.scratch_edit.setText(s.scratch_dir)
        self.budget_spin.setValue(s.mem_budget_gb)
        self.tess_edit.setText(s.tess_cache)
        self.fields_box.lambda_spin.setValue(s.lambda_th)
        self.prefix_edit.setText(s.output_prefix)
        self.pt_check.setChecked(s.parallel_triangulation)
        self.om_check.setChecked(s.ordered_merge)
        self.so_check.setChecked(s.start_over)
        self.prec_combo.setCurrentIndex(max(self.prec_combo.findData(s.precision), 0))
        # Pipeline
        self.p_nu.setCurrentText(str(p.nu))
        self.p_adaptive.setChecked(p.adaptive)
        self.p_axis.setCurrentText(p.axis)
        self.p_center.setText(p.center)
        self.p_vg.setChecked(p.vel_grad)
        self.p_pc.setCurrentIndex(max(self.p_pc.findData(p.pts_conventions), 0))
        self.p_planes.setValue(p.planes)
        self.p_thick.setValue(p.thickness)
        self.p_super.setValue(p.supersample)
        self.p_window.setText(p.window)
        self.p_grid.setCurrentText(str(p.grid))
        for name, cb in self.p_fields.items():
            cb.setChecked(name in p.fields)
        self.p_deposit.setCurrentIndex(max(self.p_deposit.findData(p.deposit), 0))
        self.p_nsub.setValue(p.nsub)
        self.p_gpu.setChecked(p.gpu)
        self.p_vm.setChecked(p.vertex_mass)
        self.p_vw.setChecked(p.volume_weighted)
        self.p_ca.setChecked(p.caustics)
        self.p_cu.setChecked(p.caustic_cusps)
        self.p_ld.setChecked(p.linear_deposit)
        self.p_pt.setChecked(p.parallel_triangulation)
        self.p_om.setChecked(p.ordered_merge)
        self.p_so.setChecked(p.start_over)
        self.p_part.setValue(p.partition)
        self.p_conc.setValue(p.max_concurrent)
        self.p_scratch.setText(p.scratch_dir)
        self.p_budget.setValue(p.mem_budget_gb)
        self.p_tess.setText(p.tess_cache)
        self.p_fields_box.lambda_spin.setValue(p.lambda_th)
        self.p_prefix.setText(p.output_prefix)
        self.p_prec.setCurrentIndex(max(self.p_prec.findData(p.precision), 0))
        self.p_force.setChecked(p.force)
        self.p_plan.setChecked(p.plan_only)
        self.p_render.setChecked(p.render)
        self.p_rfields.setText(p.render_fields)
        self.p_froot.setText("" if Path(p.figures_root) == rs.FIGURES_ROOT else p.figures_root)
        self.p_rproject.setCurrentIndex(max(self.p_rproject.findData(p.render_project), 0))
        self.p_rsmooth.setValue(p.render_smooth)
        self.p_rsmoothd.setValue(p.render_smooth_derivatives)
        self.p_rfixed.setChecked(p.render_fixed_range)
        self.p_rforce.setChecked(p.render_force)
        # Custom snapshot
        c = self.custom
        self.o_file.setText(c.input_file)
        self.o_type.setCurrentIndex(max(self.o_type.findData(c.input_type), 0))
        self.o_dim.setCurrentIndex(max(self.o_dim.findData(c.dim), 0))
        self.o_unit.setValue(c.mpc_unit)
        self.o_periodic.setChecked(c.periodic)
        self.o_alpha.setValue(c.alpha_shape)
        self.o_box.setText(" ".join(f"{v:g}" for v in c.box))
        (self.o_ps if c.estimator == "ps" else self.o_dtfe).setChecked(True)
        (self.o_lag_sep if c.lagrangian == "separate" else self.o_lag_file).setChecked(True)
        self.o_lag_path.setText(c.lagrangian_file)
        self.o_scalar.setText(c.scalar_dataset)
        self.o_grid.setCurrentText(str(c.grid))
        self.o_nsub.setValue(c.nsub)
        self.o_deposit.setCurrentIndex(max(self.o_deposit.findData(c.deposit), 0))
        self.o_gpu.setChecked(c.gpu)
        self.o_vm.setChecked(c.vertex_mass)
        self.o_vw.setChecked(c.volume_weighted)
        self.o_ca.setChecked(c.caustics)
        self.o_cu.setChecked(c.caustic_cusps)
        self.o_ld.setChecked(c.linear_deposit)
        self.o_points.setText(c.sample_points)
        self.o_pvg.setChecked(c.pts_vel_grad)
        self.o_pc.setCurrentIndex(max(self.o_pc.findData(c.pts_conventions), 0))
        self.o_window.setText(" ".join(f"{v:g}" for v in c.window))
        self.o_sdens.setCurrentIndex(max(self.o_sdens.findData(c.stream_density), 0))
        self.o_pstream.setChecked(c.per_stream)
        self.o_pids.setChecked(c.per_stream_ids)
        self.o_halo.setValue(c.halo_release)
        for name, cb in self.o_fields.items():
            cb.setChecked(name in c.fields)
        self.o_outdir.setText(c.output_dir)
        self.o_name.setText(c.output_name)
        self.o_part.setValue(c.partition)
        self.o_conc.setValue(c.max_concurrent)
        self.o_scratch.setText(c.scratch_dir)
        self.o_budget.setValue(c.mem_budget_gb)
        self.o_tess.setText(c.tess_cache)
        self.o_fields_box.lambda_spin.setValue(c.lambda_th)
        self.o_pt.setChecked(c.parallel_triangulation)
        self.o_om.setChecked(c.ordered_merge)
        self.o_prec.setCurrentIndex(max(self.o_prec.findData(c.precision), 0))
        self._own_last_file = c.input_file
        # Plots
        for i in range(self.fig_list.count()):
            it = self.fig_list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in pl.sets else Qt.Unchecked)
        self.fig_list.setCurrentRow(0)
        self.pl_method.setCurrentIndex(max(self.pl_method.findData(pl.method), 0))
        self.pl_prefix.setText(pl.prefix)
        self.pl_smooth.setValue(pl.smooth)
        self.pl_froot.setText("" if Path(pl.figures_root) == rs.FIGURES_ROOT else pl.figures_root)
        for (set_key, key), (o, w) in self.opt_widgets.items():
            self._set_opt(o, w, pl.opt(set_key, key))
        self._loading = False

    def _read_widgets(self):
        root, sim = self.root_edit.text().strip(), self.sim_combo.currentText()
        snaps = _checked(self.snap_list)
        s, p, pl = self.spec, self.pipe, self.plots
        for spec in (s, pl):
            spec.data_root, spec.sim, spec.snapshots = root, sim, list(snaps)
        p.data_root = root
        # Data
        d = self.data
        d.data_root, d.sim = root, self.d_sim.currentText().strip()
        d.snapshots = [self.d_table.item(r, 0).data(Qt.UserRole) for r in range(self.d_table.rowCount())
                       if self.d_table.item(r, 0).checkState() == Qt.Checked]
        d.download_snapshots, d.download_groupcats = self.d_dl_snap.isChecked(), self.d_dl_gc.isChecked()
        d.download_trees, d.download_ics = self.d_dl_tree.isChecked(), self.d_dl_ics.isChecked()
        d.merge_snapshots, d.merge_groupcats = self.d_mg_snap.isChecked(), self.d_mg_gc.isChecked()
        d.merge_trees, d.convert_ics = self.d_mg_tree.isChecked(), self.d_mg_ics.isChecked()
        d.delete_chunks = self.d_delete.isChecked()
        # Grids
        s.estimator = "ps" if self.ps_radio.isChecked() else "dtfe"
        s.deposit = self.deposit_combo.currentData() or "sampled"
        s.grid = _int(self.grid_combo.currentText())
        s.nsub = self.nsub_spin.value()
        s.gpu = self.gpu_check.isChecked()
        s.fields = [n for n, _ in rs.PS_FIELDS if self.field_checks[n].isChecked()]
        s.vertex_mass = self.vm_check.isChecked()
        s.volume_weighted = self.vw_check.isChecked()
        s.caustics = self.ca_check.isChecked()
        s.caustic_cusps = self.cu_check.isChecked() and self.ca_check.isChecked()   # a greyed box cannot be unticked
        s.linear_deposit = self.ld_check.isChecked()
        s.slice_plane = self.plane_combo.currentData() or ""
        s.slice_vel_grad = self.vg_check.isChecked()
        s.pts_conventions = self.pc_combo.currentData() or "grid"
        s.partition = self.part_spin.value()
        s.max_concurrent = self.conc_spin.value()
        s.scratch_dir = self.scratch_edit.text().strip()
        s.mem_budget_gb = self.budget_spin.value()
        s.tess_cache = self.tess_edit.text().strip()
        s.lambda_th = self.fields_box.lambda_spin.value()
        s.output_prefix = self.prefix_edit.text().strip()
        s.parallel_triangulation = self.pt_check.isChecked()
        s.ordered_merge = self.om_check.isChecked()
        s.start_over = self.so_check.isChecked()
        s.precision = self.prec_combo.currentData() or "single"
        # Pipeline
        p.sims = _checked(self.p_sims)
        p.nu = _int(self.p_nu.currentText())
        p.adaptive = self.p_adaptive.isChecked()
        p.axis = self.p_axis.currentText()
        p.center = self.p_center.text().strip()
        p.vel_grad = self.p_vg.isChecked()
        p.pts_conventions = self.p_pc.currentData() or "grid"
        p.planes = self.p_planes.value()
        p.thickness = self.p_thick.value()
        p.supersample = self.p_super.value()
        p.window = self.p_window.text().strip()
        p.grid = _int(self.p_grid.currentText())
        p.fields = [n for n, _ in rs.PS_FIELDS if self.p_fields[n].isChecked()]
        p.deposit = self.p_deposit.currentData() or "sampled"
        p.nsub = self.p_nsub.value()
        p.gpu = self.p_gpu.isChecked()
        p.vertex_mass = self.p_vm.isChecked()
        p.volume_weighted = self.p_vw.isChecked()
        p.caustics = self.p_ca.isChecked()
        p.caustic_cusps = self.p_cu.isChecked() and self.p_ca.isChecked()
        p.linear_deposit = self.p_ld.isChecked()
        p.parallel_triangulation = self.p_pt.isChecked()
        p.ordered_merge = self.p_om.isChecked()
        p.start_over = self.p_so.isChecked()
        p.partition = self.p_part.value()
        p.max_concurrent = self.p_conc.value()
        p.scratch_dir = self.p_scratch.text().strip()
        p.mem_budget_gb = self.p_budget.value()
        p.tess_cache = self.p_tess.text().strip()
        p.lambda_th = self.p_fields_box.lambda_spin.value()
        p.output_prefix = self.p_prefix.text().strip()
        p.precision = self.p_prec.currentData() or "single"
        p.force = self.p_force.isChecked()
        p.plan_only = self.p_plan.isChecked()
        p.render = self.p_render.isChecked()
        p.render_fields = self.p_rfields.text().strip()
        p.figures_root = self.p_froot.text().strip() or str(rs.FIGURES_ROOT)
        p.render_project = self.p_rproject.currentData() or "plane"
        p.render_smooth = self.p_rsmooth.value()
        p.render_smooth_derivatives = self.p_rsmoothd.value()
        p.render_fixed_range = self.p_rfixed.isChecked()
        p.render_force = self.p_rforce.isChecked()
        # Custom snapshot
        c = self.custom
        c.input_file = self.o_file.text().strip()
        c.input_type = int(self.o_type.currentData() or 105)
        c.dim = int(self.o_dim.currentData() or 3)
        c.mpc_unit = self.o_unit.value()
        c.periodic = self.o_periodic.isChecked()
        c.alpha_shape = self.o_alpha.value()
        try:
            c.box = [float(t) for t in self.o_box.text().replace(",", " ").split()]
        except ValueError:
            c.box = [float("nan")]          # reported by problems() as a malformed box
        c.estimator = "ps" if self.o_ps.isChecked() else "dtfe"
        c.lagrangian = "separate" if self.o_lag_sep.isChecked() else "file"
        c.lagrangian_file = self.o_lag_path.text().strip()
        c.scalar_dataset = self.o_scalar.text().strip()
        c.grid = _int(self.o_grid.currentText())
        c.nsub = self.o_nsub.value()
        c.deposit = self.o_deposit.currentData() or "sampled"
        c.gpu = self.o_gpu.isChecked()
        c.vertex_mass = self.o_vm.isChecked()
        c.volume_weighted = self.o_vw.isChecked()
        c.caustics = self.o_ca.isChecked()
        c.caustic_cusps = self.o_cu.isChecked() and self.o_ca.isChecked()
        c.linear_deposit = self.o_ld.isChecked()
        c.sample_points = self.o_points.text().strip()
        c.pts_vel_grad = self.o_pvg.isChecked()
        c.pts_conventions = self.o_pc.currentData() or "grid"
        c.stream_density = self.o_sdens.currentData() or "dtfe"
        c.per_stream = self.o_pstream.isChecked()
        c.per_stream_ids = self.o_pids.isChecked()
        c.halo_release = self.o_halo.value()
        try:
            c.window = [float(t) for t in self.o_window.text().replace(",", " ").split()]
        except ValueError:
            c.window = [float("nan")]               # unreadable: problems() says so (not silently 'no window')
        c.fields = [n for n, _ in rs.PS_FIELDS if self.o_fields[n].isChecked()]
        c.output_dir = self.o_outdir.text().strip()
        c.output_name = self.o_name.text().strip()
        c.partition = self.o_part.value()
        c.max_concurrent = self.o_conc.value()
        c.scratch_dir = self.o_scratch.text().strip()
        c.mem_budget_gb = self.o_budget.value()
        c.tess_cache = self.o_tess.text().strip()
        c.lambda_th = self.o_fields_box.lambda_spin.value()
        c.parallel_triangulation = self.o_pt.isChecked()
        c.ordered_merge = self.o_om.isChecked()
        c.precision = self.o_prec.currentData() or "single"
        # Plots
        pl.sets = _checked(self.fig_list)
        pl.method = self.pl_method.currentData() or "auto"
        pl.prefix = self.pl_prefix.text().strip()
        pl.smooth = self.pl_smooth.value()
        pl.figures_root = self.pl_froot.text().strip() or str(rs.FIGURES_ROOT)
        opts: dict[str, dict] = {}
        for (set_key, key), (o, w) in self.opt_widgets.items():
            opts.setdefault(set_key, {})[key] = self._get_opt(o, w)
        pl.options = opts
