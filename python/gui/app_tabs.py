"""MainWindow's job tabs (TabsMixin, moved from app.py): the shared data panel, and the Data, Grids, Custom
snapshot, Pipeline and Plots tabs with their option groups (presets, fields, resources, a figure set's
options) and the reactions that stay inside a tab (the Data tab's snapshot table, the deposit choice that
follows the estimator, a new custom snapshot file). The names app.py defines are read through _app at call
time; that import comes last because app.py also runs as a script: importing it again under its own name
then needs this module's class to exist already."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIntValidator
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QAbstractItemView, QHeaderView,
                               QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
                               QWidget)

import runspec as rs
import presets as P
from app_widgets import _mono, _note, _checkable, _dir_row, _exclusive, _grid_combo, _state_text


class TabsMixin:
    # ================================================================ shared data panel
    def _build_data(self) -> QWidget:
        g = QGroupBox("Data")
        f = QFormLayout(g)
        self._data_form = f
        self.root_edit = QLineEdit()
        self.root_edit.editingFinished.connect(self._root_changed)
        f.addRow("Data root", _dir_row(self.root_edit, self._browse_root))
        self.sim_combo = QComboBox()
        self.sim_combo.currentTextChanged.connect(self._sim_changed)
        f.addRow("Simulation", self.sim_combo)
        self.snap_list = QListWidget()
        self.snap_list.setFont(_mono())
        self.snap_list.setMinimumHeight(150)
        self.snap_list.setMaximumHeight(190)
        self.snap_list.itemChanged.connect(self._changed)
        f.addRow("Snapshots", self.snap_list)
        sel = QHBoxLayout()
        self.all_btn, self.none_btn, self.mark_btn = QPushButton("All"), QPushButton("None"), QPushButton()
        self.all_btn.clicked.connect(lambda: self._select_snaps("all"))
        self.none_btn.clicked.connect(lambda: self._select_snaps("none"))
        self.mark_btn.clicked.connect(lambda: self._select_snaps("mark"))
        for b in (self.all_btn, self.none_btn, self.mark_btn):
            sel.addWidget(b)
        sel.addStretch(1)
        self._snap_buttons = sel
        f.addRow("", sel)
        return g

    # ================================================================ Data tab
    def _build_data_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Simulation")
        f = QFormLayout(g)
        self.d_sim = QComboBox()
        self.d_sim.setEditable(True)
        self.d_sim.setInsertPolicy(QComboBox.NoInsert)
        self.d_sim.activated.connect(self._data_sim_changed)
        self.d_sim.lineEdit().editingFinished.connect(self._data_sim_changed)
        f.addRow("Name", self.d_sim)
        self.d_target = _note()
        f.addRow("", self.d_target)
        col.addWidget(g)

        g = QGroupBox("Snapshots")
        v = QVBoxLayout(g)
        self.d_table = QTableWidget(0, 4)
        self.d_table.setHorizontalHeaderLabels(["Snapshot", "z", "Particles", "Group catalogue"])
        self.d_table.verticalHeader().setVisible(False)
        self.d_table.verticalHeader().setDefaultSectionSize(22)
        self.d_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.d_table.setSelectionMode(QAbstractItemView.NoSelection)
        head = self.d_table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.Stretch)
        for c in (0, 1):                            # snapshot number and z: as wide as they need
            head.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.d_table.setMinimumHeight(230)
        self.d_table.itemChanged.connect(self._changed)
        v.addWidget(self.d_table, 1)        # grows with the window (else ten rows above empty space)
        sel = QHBoxLayout()
        for label, mode in (("All", "all"), ("None", "none"), ("Not merged", "missing")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, m=mode: self._select_data(m))
            sel.addWidget(b)
        self.d_other = QLineEdit(placeholderText="add snapshots, e.g. 59 72")
        self.d_other.editingFinished.connect(self._add_data_snapshots)
        sel.addWidget(self.d_other, 1)
        v.addLayout(sel)
        self.d_whole = _note()
        v.addWidget(self.d_whole)
        col.addWidget(g, 1)

        g = QGroupBox("Download from the TNG API")
        v = QVBoxLayout(g)
        self.d_dl_snap = QCheckBox("Particle snapshots")
        self.d_dl_gc = QCheckBox("Group catalogues")
        self.d_dl_tree = QCheckBox("Merger trees (whole simulation)")
        self.d_dl_ics = QCheckBox("Initial conditions (whole simulation)")
        for cb, tip in ((self.d_dl_snap, "For the ticked snapshots"),
                        (self.d_dl_gc, "For the ticked snapshots"),
                        (self.d_dl_tree, "One set for the whole simulation"),
                        (self.d_dl_ics, "One set for the whole simulation")):
            cb.setToolTip(tip)
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        col.addWidget(g)

        g = QGroupBox("Merge into combined files")
        v = QVBoxLayout(g)
        self.d_mg_snap = QCheckBox("Snapshots")
        self.d_mg_gc = QCheckBox("Group catalogues")
        self.d_mg_tree = QCheckBox("Merger trees")
        self.d_mg_ics = QCheckBox("Initial conditions")
        self.d_delete = QCheckBox("Delete the raw chunks after a verified merge")
        for cb, tip in ((self.d_mg_snap, "Writes combined_NNN.hdf5 (h-free units)"),
                        (self.d_mg_gc, "Writes combined_fof_subhalo_tab_NNN.hdf5"),
                        (self.d_mg_tree, "Writes combined_tree_extended.hdf5"),
                        (self.d_mg_ics, "Writes combined_ics.hdf5"),
                        (self.d_delete, "The chunks are the only copy of the fields the merge drops "
                                        "(potential, Subfind densities)")):
            cb.setToolTip(tip)
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        col.addWidget(g)
        # (no trailing stretch: the snapshot table above takes the room a taller window gives)
        w = QWidget()
        w.setLayout(col)
        return w

    def _data_sim_changed(self, *_):
        if self._loading:
            return
        self.data.sim = self.d_sim.currentText().strip()
        self.data.snapshots = []
        self._fill_data_table()
        self._changed()

    def _fill_data_table(self):
        was = self._loading
        self._loading = True
        sim, root = self.d_sim.currentText().strip(), self.root_edit.text().strip()
        wanted = set(self.data.snapshots)
        self.d_table.setRowCount(0)
        self.d_target.setText("")
        self.d_whole.setText("")
        if sim and rs.root_ok(root):
            sp = rs.sim_dir(sim, root)
            n_part = rs.dm_particles(sim)
            size = (f"  ·  {round(n_part ** (1 / 3))}³ particles, {n_part * rs.MERGED_BYTES_PER_PARTICLE / 1e9:.1f} GB "
                    f"per snapshot") if n_part else ""
            try:
                where = str(sp.relative_to(Path(root)))
            except ValueError:
                where = str(sp)
            self.d_target.setText(where + ("  (new folder)" if not sp.is_dir() else "") + size)
            self.d_target.setToolTip(f"{sp}\n~{n_part * rs.RAW_BYTES_PER_PARTICLE / 1e9:.1f} GB to download, "
                                     f"{n_part * rs.MERGED_BYTES_PER_PARTICLE / 1e9:.1f} GB merged, per snapshot"
                                     if n_part else str(sp))
            for n in sorted(set(rs.data_snapshots(sp)) | wanted):
                st = rs.snapshot_state(sp, n)
                z = rs.SNAPSHOT_LADDER.get(n)
                if z is None and st["merged"]:
                    z = rs._redshift(sp / f"snapdir_{n:03d}" / f"combined_{n:03d}.hdf5")
                row = self.d_table.rowCount()
                self.d_table.insertRow(row)
                first = QTableWidgetItem(f"{n:03d}")
                first.setData(Qt.UserRole, n)
                first.setData(Qt.UserRole + 1, st["merged"])
                first.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                first.setCheckState(Qt.Checked if n in wanted else Qt.Unchecked)
                cells = [first, QTableWidgetItem("?" if z is None else f"{z:.2f}"),
                         QTableWidgetItem(_state_text(st["chunks"], st["merged"])),
                         QTableWidgetItem(_state_text(st["gc_chunks"], st["gc_merged"]))]
                for col, (item, (chunks, merged)) in enumerate(zip(
                        cells, [(0, False), (0, False), (st["chunks"], st["merged"]),
                                (st["gc_chunks"], st["gc_merged"])])):
                    if col >= 2:
                        item.setForeground(_app.MERGED_COLOR if merged else _app.CHUNKS_COLOR if chunks else QColor("gray"))
                    if col:
                        item.setFlags(Qt.ItemIsEnabled)
                    self.d_table.setItem(row, col, item)
            w = rs.sim_state(sp)
            self.d_whole.setText(
                f"Merger trees: {_state_text(w['tree_chunks'], w['tree_merged'])}   ·   Initial conditions: "
                + ("converted" if w["ics_converted"] else "downloaded, not converted" if w["ics_raw"] else "—"))
        self._loading = was

    def _select_data(self, mode: str):
        self._loading = True
        for row in range(self.d_table.rowCount()):
            it = self.d_table.item(row, 0)
            on = mode == "all" or (mode == "missing" and not it.data(Qt.UserRole + 1))
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._loading = False
        self._changed()

    def _add_data_snapshots(self):
        extra = {int(t) for t in self.d_other.text().replace(",", " ").split() if t.isdigit() and int(t) <= 99}
        if not extra:
            return
        self.d_other.clear()
        self._read_widgets()
        self.data.snapshots = sorted(set(self.data.snapshots) | extra)
        self._fill_data_table()
        self._changed()

    # ================================================================ presets
    def _preset_group(self, target: str) -> QGroupBox:
        """'Start from': the built-in and saved presets for the Grids ('grids') or Custom snapshot ('custom') job."""
        g = QGroupBox("Start from a preset")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        combo = QComboBox()
        combo.setToolTip("Sets the grid, fields and options below; everything stays editable")
        save = QPushButton("Save as preset…")
        delete = QPushButton("Delete")
        row.addWidget(combo, 1)
        row.addWidget(save)
        row.addWidget(delete)
        v.addLayout(row)
        about = _note()
        about.hide()
        v.addWidget(about)
        combo.activated.connect(lambda _i: self._apply_preset(target))
        save.clicked.connect(lambda: self._save_preset(target))
        delete.clicked.connect(lambda: self._delete_preset(target))
        setattr(self, f"_preset_{target}", (combo, delete, about))
        self._fill_presets(target)
        return g

    def _fill_presets(self, target: str):
        combo, delete, about = getattr(self, f"_preset_{target}")
        keep = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("choose…", "")
        for name in P.preset_names(self.user_presets):
            combo.addItem(name, name)
        combo.setCurrentIndex(max(combo.findData(keep), 0))
        combo.blockSignals(False)
        delete.setEnabled(bool(combo.currentData()) and combo.currentData() not in P.BUILTIN_PRESETS)
        about.setText("")
        about.hide()

    def _apply_preset(self, target: str):
        combo, delete, about = getattr(self, f"_preset_{target}")
        name = combo.currentData()
        delete.setEnabled(bool(name) and name not in P.BUILTIN_PRESETS)
        found = P.preset(name, self.user_presets) if name else None
        if not found:
            about.setText("")
            about.hide()
            return
        text, settings = found
        self._read_widgets()
        spec = self.spec if target == "grids" else self.custom
        planes = rs.planes(self._sim_path()) if (target == "grids" and self._sim_path()) else []
        changed = P.apply_preset(spec, settings, planes)
        if "gpu" not in settings:          # the preset leaves the GPU to the run's size (see recommended_gpu)
            n = (rs.dm_particles(spec.sim) if target == "grids" else
                 rs.hdf5_particle_count(Path(spec.input_file).expanduser()) if spec.input_file else None)
            rec = rs.recommended_gpu(spec.deposit, spec.fields, n, spec.estimator, spec.precision)
            if spec.gpu != rec:
                spec.gpu = rec
                changed.append("gpu")
        if target == "grids" and settings.get("slice_plane") == "largest" and not planes:
            text += "  (no image plane for this simulation yet: make one with the Pipeline tab)"
        about.setText(text)
        about.show()
        self._load_into_widgets()
        self._changed()
        self.status_message(f"preset '{name}': {len(changed)} setting{'s' if len(changed) != 1 else ''} changed")

    def _save_preset(self, target: str):
        name, ok = QInputDialog.getText(self, "Save as preset", "Name for the current settings:")
        name = name.strip()
        if not ok or not name:
            return
        if name in P.BUILTIN_PRESETS:
            QMessageBox.information(self, "Save as preset", f"'{name}' is a built-in preset; choose another name.")
            return
        self._read_widgets()
        self.user_presets[name] = P.capture(self.spec if target == "grids" else self.custom)
        self._save()
        for t in ("grids", "custom"):
            self._fill_presets(t)
        combo = getattr(self, f"_preset_{target}")[0]
        combo.setCurrentIndex(combo.findData(name))
        self.status_message(f"saved preset '{name}'")

    def _delete_preset(self, target: str):
        combo = getattr(self, f"_preset_{target}")[0]
        name = combo.currentData()
        if name and name in self.user_presets:
            del self.user_presets[name]
            self._save()
            for t in ("grids", "custom"):
                self._fill_presets(t)

    # ================================================================ Grids tab
    def _build_grids_tab(self) -> QWidget:
        col = QVBoxLayout()
        col.addWidget(self._preset_group("grids"))
        g = QGroupBox("Estimator and grid")
        f = QFormLayout(g)
        est = QHBoxLayout()
        self.ps_radio = QRadioButton("PS-DTFE (phase space)")
        self.dtfe_radio = QRadioButton("DTFE (standard)")
        grp = QButtonGroup(self)
        grp.addButton(self.ps_radio)
        grp.addButton(self.dtfe_radio)
        # before _changed (slots run in connection order): the deposit choice follows the estimator
        self.ps_radio.toggled.connect(lambda on: self._estimator_switched(self.deposit_combo, on))
        self.ps_radio.toggled.connect(self._changed)
        est.addWidget(self.ps_radio)
        est.addWidget(self.dtfe_radio)
        est.addStretch(1)
        f.addRow("Estimator", est)
        self.deposit_combo = QComboBox()
        self.deposit_combo.addItem(P.label("sampled"), "sampled")
        self.deposit_combo.addItem(P.label("exact"), "exact")
        self.deposit_combo.setItemData(0, P.tooltip("sampled"), Qt.ToolTipRole)
        self.deposit_combo.setItemData(1, P.tooltip("exact"), Qt.ToolTipRole)
        # the estimator from the radio, not the spec: an estimator switch changes the combo before _read_widgets
        self.deposit_combo.currentIndexChanged.connect(
            lambda _i: self._deposit_changed(self.deposit_combo, self.gpu_check,
                                             "ps" if self.ps_radio.isChecked() else "dtfe"))
        f.addRow("Deposit", self.deposit_combo)
        gr = QHBoxLayout()
        self.grid_combo = _grid_combo(self)
        self.grid_combo.currentTextChanged.connect(self._changed)
        gr.addWidget(self.grid_combo)
        gr.addWidget(QLabel("³ cells    sub-samples per axis"))
        self.nsub_spin = QSpinBox(minimum=1, maximum=6)
        self.nsub_spin.valueChanged.connect(self._changed)
        gr.addWidget(self.nsub_spin)
        gr.addStretch(1)
        f.addRow("Grid", gr)
        self.gpu_check = QCheckBox(P.label("gpu"))
        self.gpu_check.setToolTip(P.tooltip("gpu"))
        self.gpu_check.toggled.connect(self._changed)
        f.addRow("", self.gpu_check)
        col.addWidget(g)

        g, self.field_checks = self._fields_group()
        self.fields_box = g
        self.fields_note = _note()
        g.layout().addWidget(self.fields_note, g.layout().rowCount(), 0, 1, 3)      # below the threshold row
        col.addWidget(g)

        self.ps_group = QGroupBox("Phase-space options")
        v = QVBoxLayout(self.ps_group)
        self.vm_check = QCheckBox(P.label("vertex_mass"))
        self.vw_check = QCheckBox(P.label("volume_weighted"))
        self.ca_check = QCheckBox(P.label("caustics"))
        self.cu_check = QCheckBox(P.label("caustic_cusps"))
        self.ld_check = QCheckBox(P.label("linear_deposit"))
        for cb, key in ((self.vm_check, "vertex_mass"), (self.vw_check, "volume_weighted"),
                        (self.ca_check, "caustics"), (self.cu_check, "caustic_cusps"), (self.ld_check, "linear_deposit")):
            cb.setToolTip(P.tooltip(key))
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        _exclusive(self.ld_check, self.vw_check)
        col.addWidget(self.ps_group)

        self.slice_group = QGroupBox("Hi-res slice (point evaluation)")
        f = QFormLayout(self.slice_group)
        self.plane_combo = QComboBox()
        self.plane_combo.currentIndexChanged.connect(self._changed)
        f.addRow("Sample points", self.plane_combo)
        self.vg_check = QCheckBox(P.label("slice_vel_grad"))
        self.vg_check.setToolTip(P.tooltip("slice_vel_grad"))
        self.vg_check.toggled.connect(self._changed)
        f.addRow("", self.vg_check)
        self.pc_combo = _app._pts_conventions_combo(self._changed)       # PS-DTFE: the plane's conventions
        f.addRow(P.label("pts_conventions"), self.pc_combo)
        col.addWidget(self.slice_group)

        g, (self.part_spin, self.conc_spin, self.scratch_edit, self.prefix_edit,
            self.pt_check, self.prec_combo, self.budget_spin, self.tess_edit, self.om_check,
            self.so_check) = self._resources_group()
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _deposit_changed(self, combo: QComboBox, gpu: QCheckBox, estimator: str = "ps", dim: int = 3):
        """Picking the exact deposit ticks the GPU (PS-DTFE: 2x faster than the threaded CPU deposit at
        0.26M particles, 29 s against 57 s on a 256³ grid, and more at larger sizes; standard DTFE's exact
        cell average: 6.6x, 4.1 s against 26.9 s); the box stays editable. A 2D run has no GPU."""
        if not self._loading and combo.currentData() == "exact" and rs.gpu_built(estimator, dim=dim) and not gpu.isChecked():   # (either pair has the GPU when the single one has)
            gpu.setChecked(True)            # (its toggled signal runs _changed)
            self.status_message("exact deposit: the GPU is ticked (2x faster than the CPU at 0.26M particles, "
                                "more at larger sizes)" if estimator == "ps" else
                                "exact cell averages: the GPU is ticked (6.6x faster than the CPU at 0.26M particles)")
        self._changed()

    def _estimator_switched(self, combo: QComboBox, ps_now: bool):
        """An estimator switch by the user moves the deposit choice to the new estimator's production setting
        (rs.production_deposit): PS-DTFE the sampled deposit, standard DTFE the exact cell average,
        run_dtfe.sh's default; keeping 'sampled' would run a standard job on the biased Monte-Carlo average
        unless it was changed by hand. (With two choices, 'a choice at the old estimator's production setting
        follows to the new one's' is the same rule.) The other choice stays selectable afterwards. A choice made
        for an estimator comes back with it (per tab, this session): else PS-DTFE's exact deposit would be lost on
        the way PS -> standard -> PS, the switch back giving the sampled one."""
        if self._loading:
            return
        new, old = ("ps", "dtfe") if ps_now else ("dtfe", "ps")
        kept = self._deposit_kept.setdefault("grids" if combo is self.deposit_combo else "own", {})
        if combo.currentData():
            kept[old] = combo.currentData()          # the estimator left: its choice, for the way back
        want = kept.get(new) or rs.production_deposit(new)
        if combo.currentData() != want:
            gpu = self.gpu_check if combo is self.deposit_combo else self.o_gpu
            was = gpu.isChecked()
            combo.setCurrentIndex(max(combo.findData(want), 0))      # its signal: the GPU tick, _changed
            on_gpu = ", on the GPU (6.6x faster than the CPU)" if gpu.isChecked() and not was else ""
            if want != rs.production_deposit(new) and ps_now:            # the user's own choice back
                self.status_message("PS-DTFE: the exact deposit again, as chosen for it before"
                                    + (", on the GPU (2x faster than the CPU)" if gpu.isChecked() and not was else ""))
            elif want != rs.production_deposit(new):
                self.status_message("standard DTFE: the Monte-Carlo average again, as chosen for it before; the "
                                    "exact cell average stays selectable")
            else:
                self.status_message(f"standard DTFE: the exact cell average, its production setting{on_gpu}; the "
                                    "Monte-Carlo average stays selectable" if not ps_now else
                                    "PS-DTFE: the sampled deposit, its production setting")

    @staticmethod
    def _deposit_texts(combo: QComboBox, ps: bool):
        """the deposit choice in the words of the estimator it applies to (the same two settings)"""
        for i, key in enumerate(("sampled", "exact")):
            k = key if ps else key + "_dtfe"
            if combo.itemText(i) != P.label(k):
                combo.setItemText(i, P.label(k))
                combo.setItemData(i, P.tooltip(k) if ps else P.tooltip(k, "DTFE"), Qt.ToolTipRole)

    def _fields_group(self, title: str = "Fields") -> tuple[QGroupBox, dict[str, QCheckBox]]:
        g = QGroupBox(title)
        grid = QGridLayout(g)
        checks: dict[str, QCheckBox] = {}
        for i, (name, label) in enumerate(rs.PS_FIELDS):
            cb = QCheckBox(label)
            cb.toggled.connect(self._changed)
            checks[name] = cb
            grid.addWidget(cb, i // 3, i % 3)
        for c in range(3):                  # equal columns: the last one must not be starved of width
            grid.setColumnStretch(c, 1)
        row = QHBoxLayout()                 # the T-web / V-web eigenvalue threshold: only with a web class ticked
        lam = QDoubleSpinBox(minimum=0.0, maximum=10.0, decimals=2, singleStep=0.05, value=rs.LAMBDA_TH_DEFAULT)
        lam.setToolTip("λ_th of the T-web / V-web classes: an eigenvalue above it counts as collapsing (a void "
                       "has none, a wall one, a filament two, a node three). 0.3 is the scripts' default; the "
                       "binary alone would use 0.")
        lam.valueChanged.connect(self._changed)
        row.addWidget(QLabel("T-web / V-web threshold λ_th"))
        row.addWidget(lam)
        row.addStretch(1)
        grid.addLayout(row, (len(rs.PS_FIELDS) + 2) // 3, 0, 1, 3)
        g.lambda_spin = lam

        def web_on(_on=False):
            lam.setEnabled(any(checks[w].isChecked() and checks[w].isEnabled() for w in rs.WEB_FIELDS))
        for w in rs.WEB_FIELDS:
            checks[w].toggled.connect(web_on)
        g.sync_lambda = web_on
        web_on()
        return g, checks

    def _resources_group(self):
        g = QGroupBox("Resources and output")
        f = QFormLayout(g)
        part = QSpinBox(minimum=0, maximum=8)
        part.setSpecialValueText("auto")
        part.setSuffix("³ partitions")
        part.valueChanged.connect(self._changed)
        f.addRow("Partition", part)
        conc = QSpinBox(minimum=0, maximum=16)
        conc.setSpecialValueText("auto")
        conc.valueChanged.connect(self._changed)
        f.addRow("Concurrent partitions", conc)
        scratch = QLineEdit()
        scratch.setPlaceholderText("none: grids in RAM")
        scratch.editingFinished.connect(self._changed)
        f.addRow("Scratch directory", _dir_row(scratch, lambda: self._browse_scratch(scratch)))
        ptri = QCheckBox(P.label("parallel_triangulation"))
        ptri.setToolTip(P.tooltip("parallel_triangulation"))
        ptri.toggled.connect(self._changed)
        f.addRow("", ptri)
        # K1: the partitions merged in index order, so two identical runs give bit-identical grids
        ordered = QCheckBox(P.label("ordered_merge"))
        ordered.setToolTip(P.tooltip("ordered_merge"))
        ordered.toggled.connect(self._changed)
        f.addRow("", ordered)
        # a killed partitioned run's checkpoint is resumed by the next run of the same job; this starts
        # it over instead (the only way out when the resume is refused: other settings, an incomplete checkpoint).
        # The Custom tab hides it: a Custom run never resumes.
        start_over = QCheckBox("Start over: do not resume a killed run's checkpoint")
        start_over.setToolTip("A partitioned run that was killed keeps a checkpoint of its finished partitions, and the "
                              "next run of the same job continues from it. Ticked, the run starts from the beginning "
                              "and replaces the checkpoint (PS_RESUME=0; standard DTFE: DTFE_RESUME=0): needed when "
                              "the settings changed since, as the program then refuses to resume.")
        start_over.toggled.connect(self._changed)
        f.addRow("", start_over)
        prec = QComboBox()
        prec.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)   # its long double-precision text
        prec.setMinimumContentsLength(24)       # else the tab gets wider than the pane: the '…' buttons get cut off
        for key in rs.PRECISIONS:
            prec.addItem(rs.PRECISION_TEXT[key], key)
        prec.setToolTip("Which pair of programs runs: the single-precision one (the default), or the "
                        "double-precision pair, which keeps every position, density and field in float64 from "
                        "the input read onward (TNG stores its coordinates in double) and writes float64 grids, "
                        "at about twice the memory. Setup builds the pair.")
        prec.currentIndexChanged.connect(self._changed)
        f.addRow("Precision", prec)
        prefix = QLineEdit()
        prefix.editingFinished.connect(self._changed)
        f.addRow("Output prefix", prefix)
        budget = QDoubleSpinBox(minimum=0.0, maximum=4096.0, decimals=0, singleStep=4.0)
        budget.setSpecialValueText("auto (free memory)")
        budget.setSuffix(" GB")
        budget.setToolTip("The memory the auto-tuner plans the partitions and concurrency for "
                          "(DTFE_MEM_BUDGET_GB). Auto: what is free when the run starts. Set it lower to leave room "
                          "for other work, higher when you know the memory will be free (it may swap).")
        budget.valueChanged.connect(self._changed)
        f.addRow("Memory budget", budget)
        tess = QLineEdit()
        tess.setPlaceholderText("none: triangulate every run")
        tess.setToolTip("Reuse the Delaunay tessellation of an identical earlier run (the binary's tessellation "
                        "cache): the "
                        "dominant fixed cost of a rerun with other fields, grids or sample points (measured "
                        "10.9 s -> 4.6 s). A LOCAL folder (iCloud is refused); about 0.15 KB per vertex of disk.")
        tess.editingFinished.connect(self._changed)
        f.addRow("Tessellation cache", _dir_row(tess, lambda: self._browse_dir(tess, "Tessellation cache folder")))
        return g, (part, conc, scratch, prefix, ptri, prec, budget, tess, ordered, start_over)

    # ================================================================ Custom snapshot tab
    def _build_own_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Snapshot")
        f = QFormLayout(g)
        self.o_file = QLineEdit(placeholderText="choose a snapshot file")
        self.o_file.editingFinished.connect(self._own_file_changed)
        f.addRow("File", _dir_row(self.o_file, self._browse_own_file))
        self.o_type = QComboBox()
        for t, text in rs.INPUT_TYPES.items():
            self.o_type.addItem(text, t)
        self.o_type.currentIndexChanged.connect(self._changed)
        f.addRow("Format", self.o_type)
        self.o_dim = QComboBox()
        for d in rs.DIMS:
            self.o_dim.addItem(rs.DIM_TEXT[d], d)
        self.o_dim.setToolTip("3D, or 2D for a snapshot in a plane (its coordinates have two columns): the 2D "
                              "programs DTFE-2d and PS-DTFE-2d run it, on the CPU, with every field (the 2D "
                              "T-web classes: 0 void, 1 filament, 2 node). Set from the file when you choose one "
                              "(HDF5 and text files say which).")
        self.o_dim.currentIndexChanged.connect(self._changed)
        f.addRow("Dimensions", self.o_dim)
        un = QHBoxLayout()
        self.o_unit = QDoubleSpinBox(minimum=1e-6, maximum=1e9, decimals=4, value=1.0)
        self.o_unit.setToolTip("How many of the file's length units make one Mpc: 1 when the coordinates "
                               "are in Mpc, 1000 in kpc (e.g. Gadget's usual kpc/h: 1000, and h then stays "
                               "in the lengths)")
        self.o_unit.valueChanged.connect(self._changed)
        un.addWidget(QLabel("1 Mpc ="))
        un.addWidget(self.o_unit)
        un.addWidget(QLabel("file units"))
        for text, value in (("Mpc", 1.0), ("kpc", 1000.0)):
            b = QToolButton(text=text)
            b.clicked.connect(lambda _=False, v=value: self.o_unit.setValue(v))
            un.addWidget(b)
        un.addStretch(1)
        f.addRow("Length unit", un)
        self.o_periodic = QCheckBox(P.label("periodic"))
        self.o_periodic.setToolTip(P.tooltip("periodic", "DTFE"))
        self.o_periodic.toggled.connect(self._changed)
        f.addRow("", self.o_periodic)
        self.o_alpha = QDoubleSpinBox(minimum=0.0, maximum=50.0, decimals=1, singleStep=0.5, value=3.0)
        self.o_alpha.setSuffix(" mean particle spacings")
        self.o_alpha.setSpecialValueText("whole convex hull")
        self.o_alpha.setToolTip(P.tooltip("alpha_shape"))
        self.o_alpha.valueChanged.connect(self._changed)
        f.addRow("Alpha shape", self.o_alpha)
        self.o_snap_form = f
        self.o_box = QLineEdit(placeholderText="from the file header (Gadget BoxSize)")
        self.o_box.setToolTip("Optional: xlo xhi ylo yhi zlo zhi in the file's units (all >= 0)")
        self.o_box.editingFinished.connect(self._changed)
        f.addRow("Box", self.o_box)
        col.addWidget(g)

        g = QGroupBox("Estimator")
        f = QFormLayout(g)
        est = QHBoxLayout()
        self.o_ps = QRadioButton("PS-DTFE (phase space: streams, caustics)")
        self.o_dtfe = QRadioButton("DTFE (standard)")
        grp = QButtonGroup(self)
        grp.addButton(self.o_ps)
        grp.addButton(self.o_dtfe)
        self.o_ps.toggled.connect(lambda on: self._estimator_switched(self.o_deposit, on))   # (the Grids tab's rule)
        self.o_ps.toggled.connect(self._changed)
        est.addWidget(self.o_ps)
        est.addWidget(self.o_dtfe)
        est.addStretch(1)
        f.addRow("", est)
        lg = QVBoxLayout()
        self.o_lag_file = QRadioButton("Initial positions stored in the snapshot ('InitialCoordinates')")
        self.o_lag_sep = QRadioButton("Initial positions in a separate snapshot (matched by particle ID):")
        grp2 = QButtonGroup(self)
        grp2.addButton(self.o_lag_file)
        grp2.addButton(self.o_lag_sep)
        self.o_lag_file.toggled.connect(self._changed)
        lg.addWidget(self.o_lag_file)
        lg.addWidget(self.o_lag_sep)
        self.o_lag_path = QLineEdit(placeholderText="the initial-conditions snapshot (Gadget HDF5)")
        self.o_lag_path.editingFinished.connect(self._changed)
        lg.addLayout(_dir_row(self.o_lag_path, lambda: self._browse_file(self.o_lag_path, "Initial conditions")))
        self.o_lag_box = QWidget()
        self.o_lag_box.setLayout(lg)
        f.addRow("Lagrangian", self.o_lag_box)
        self.o_scalar = QLineEdit(placeholderText="none (e.g. Potential)")
        self.o_scalar.setToolTip(P.tooltip("scalar"))
        self.o_scalar.editingFinished.connect(self._changed)
        f.addRow("Per-particle value", self.o_scalar)
        col.addWidget(g)

        col.addWidget(self._preset_group("custom"))
        g = QGroupBox("Grid and fields")
        f = QFormLayout(g)
        gr = QHBoxLayout()
        self.o_grid = _grid_combo(self)
        self.o_grid.currentTextChanged.connect(self._changed)
        gr.addWidget(self.o_grid)
        self.o_grid_cells = QLabel("³ cells    sub-samples per axis")
        gr.addWidget(self.o_grid_cells)
        self.o_nsub = QSpinBox(minimum=1, maximum=6)
        self.o_nsub.valueChanged.connect(self._changed)
        gr.addWidget(self.o_nsub)
        gr.addStretch(1)
        f.addRow("Grid", gr)
        self.o_deposit = QComboBox()
        self.o_deposit.addItem(P.label("sampled"), "sampled")
        self.o_deposit.addItem(P.label("exact"), "exact")
        self.o_deposit.setItemData(0, P.tooltip("sampled"), Qt.ToolTipRole)
        self.o_deposit.setItemData(1, P.tooltip("exact"), Qt.ToolTipRole)
        self.o_deposit.currentIndexChanged.connect(lambda _i: self._deposit_changed(self.o_deposit, self.o_gpu,
                                                                                    "ps" if self.o_ps.isChecked()
                                                                                    else "dtfe", self.custom.dim))
        f.addRow("Deposit", self.o_deposit)
        self.o_gpu = QCheckBox(P.label("gpu"))
        self.o_gpu.setToolTip(P.tooltip("gpu"))
        self.o_vm = QCheckBox(P.label("vertex_mass"))
        self.o_vm.setToolTip(P.tooltip("vertex_mass"))
        self.o_vw = QCheckBox(P.label("volume_weighted"))
        self.o_vw.setToolTip(P.tooltip("volume_weighted"))
        self.o_ca = QCheckBox(P.label("caustics"))
        self.o_ca.setToolTip(P.tooltip("caustics"))
        self.o_cu = QCheckBox(P.label("caustic_cusps"))
        self.o_cu.setToolTip(P.tooltip("caustic_cusps"))
        self.o_ld = QCheckBox(P.label("linear_deposit"))
        self.o_ld.setToolTip(P.tooltip("linear_deposit"))
        for cb in (self.o_gpu, self.o_vm, self.o_vw, self.o_ca, self.o_cu, self.o_ld):
            cb.toggled.connect(self._changed)
            f.addRow("", cb)
        _exclusive(self.o_ld, self.o_vw)
        col.addWidget(g)
        g = QGroupBox("Points and window")
        f = QFormLayout(g)
        self.o_points = QLineEdit(placeholderText="none: the grid only")
        self.o_points.setToolTip(P.tooltip("sample_points"))
        self.o_points.editingFinished.connect(self._changed)
        f.addRow("Sample points", _dir_row(self.o_points, lambda: (self._browse_file(self.o_points, "Sample points file"),
                                                                    self._changed())))
        self.o_pvg = QCheckBox(P.label("slice_vel_grad"))
        self.o_pvg.setToolTip(P.tooltip("slice_vel_grad"))
        self.o_pvg.toggled.connect(self._changed)
        f.addRow("", self.o_pvg)
        self.o_pc = _app._pts_conventions_combo(self._changed)           # PS-DTFE: the points' conventions
        f.addRow(P.label("pts_conventions"), self.o_pc)
        self.o_window = QLineEdit(placeholderText="none: the whole grid")
        self.o_window.setToolTip(P.tooltip("window") + "\n\nGive x0 x1 y0 y1 z0 z1 in Mpc (2D: four numbers).")
        self.o_window.editingFinished.connect(self._changed)
        f.addRow("Window (Mpc)", self.o_window)
        self.o_sdens = QComboBox()
        self.o_sdens.addItem("DTFE interpolation of the vertex densities (default)", "dtfe")
        self.o_sdens.addItem("geometric: the tetrahedron's mass over its volume", "geometric")
        self.o_sdens.setToolTip(P.tooltip("stream_density"))
        self.o_sdens.currentIndexChanged.connect(self._changed)
        f.addRow("Stream density", self.o_sdens)
        self.o_pstream = QCheckBox(P.label("per_stream"))
        self.o_pstream.setToolTip(P.tooltip("per_stream"))
        self.o_pids = QCheckBox(P.label("per_stream_ids"))
        self.o_pids.setToolTip(P.tooltip("per_stream_ids"))
        for cb in (self.o_pstream, self.o_pids):
            cb.toggled.connect(self._changed)
            f.addRow("", cb)
        hr = QHBoxLayout()
        self.o_halo = QDoubleSpinBox(minimum=0.0, maximum=1e7, decimals=0, singleStep=100.0)
        self.o_halo.setSpecialValueText("off")
        self.o_halo.setToolTip(P.tooltip("halo_release"))
        self.o_halo.valueChanged.connect(self._changed)
        hr.addWidget(self.o_halo)
        hr.addStretch(1)
        f.addRow("Halo release (ρ/ρ̄)", hr)
        self.o_points_box = g
        col.addWidget(g)
        g, self.o_fields = self._fields_group()
        self.o_fields_box = g
        col.addWidget(g)

        g = QGroupBox("More options")
        f = QFormLayout(g)
        self.o_extra = QLineEdit(placeholderText="options the tab has no control for, as on a command line")
        self.o_extra.setToolTip("Any of the program's options the tab has no control for, typed as on a command "
                                "line (see the program's help): they go after the tab's own, and an option the "
                                "tab sets already is reported, not passed twice.")
        self.o_extra.editingFinished.connect(self._changed)
        f.addRow("Extra options", self.o_extra)
        col.addWidget(g)

        g = QGroupBox("Output")
        f = QFormLayout(g)
        self.o_outdir = QLineEdit()
        self.o_outdir.editingFinished.connect(self._changed)
        f.addRow("Folder", _dir_row(self.o_outdir, lambda: self._browse_dir(self.o_outdir, "Output folder")))
        self.o_name = QLineEdit(placeholderText="the snapshot's file name")
        self.o_name.editingFinished.connect(self._changed)
        f.addRow("Name", self.o_name)
        col.addWidget(g)
        g, (self.o_part, self.o_conc, self.o_scratch, _prefix, self.o_pt, self.o_prec,
            self.o_budget, self.o_tess, self.o_om, _start_over) = self._resources_group()
        for wid in (_prefix, _start_over):
            wid.hide()
            lab = g.layout().labelForField(wid)     # an addRow("", ...) row may have none
            if lab is not None:
                lab.hide()
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _browse_file(self, edit: QLineEdit, title: str):
        start = edit.text() or str(Path.home())
        f, _ = QFileDialog.getOpenFileName(self, title, start)
        if f:
            edit.setText(f)
            self._changed()

    def _browse_dir(self, edit: QLineEdit, title: str):
        d = QFileDialog.getExistingDirectory(self, title, edit.text() or str(Path.home()))
        if d:
            edit.setText(d)
            self._changed()

    def _browse_own_file(self):
        self._browse_file(self.o_file, "Your snapshot file")
        self._own_file_changed()

    def _own_file_changed(self):
        """A new file: guess its format from the name and tick the GPU when its particle count says
        the GPU deposit is the faster one (the user can still change both)."""
        f = self.o_file.text().strip()
        if not f or f == self._own_last_file:
            return
        self._own_last_file = f
        suffix = Path(f).suffix.lower()
        guess = 105 if suffix in (".hdf5", ".h5", ".hdf") else 111 if suffix in (".txt", ".dat", ".csv") else None
        if guess is not None:
            self.o_type.setCurrentIndex(max(self.o_type.findData(guess), 0))
        dim = rs.snapshot_dim(Path(f).expanduser(), guess) if guess is not None else None
        if dim is not None:                 # the file says: 2D or 3D
            self.o_dim.setCurrentIndex(max(self.o_dim.findData(dim), 0))
        if guess == 105 and Path(f).expanduser().is_file():
            self._read_widgets()
            c = self.custom
            n = rs.hdf5_particle_count(Path(f).expanduser())
            self.o_gpu.setChecked(rs.recommended_gpu(c.deposit, c.fields, n, c.estimator, c.precision, c.dim))
        self._changed()

    # ================================================================ Pipeline tab
    def _build_pipeline_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Simulations")
        v = QVBoxLayout(g)
        self.p_sims = QListWidget()
        self.p_sims.setMaximumHeight(100)
        self.p_sims.itemChanged.connect(self._changed)
        self.p_sims.setDragDropMode(QAbstractItemView.InternalMove)     # the run order: drag a simulation up or down
        self.p_sims.setToolTip("Tick the simulations to run. They run from the top down: drag one to change the order")
        self.p_sims.model().rowsMoved.connect(lambda *_: self._changed())
        v.addWidget(self.p_sims)
        v.addWidget(_note("Computes every snapshot whose slice is missing or out of date."))
        col.addWidget(g)

        g = QGroupBox("Image plane")
        g.setToolTip("A full-box slice, evaluated at every pixel (point evaluation)")
        f = QFormLayout(g)
        self.p_nu = QComboBox()
        self.p_nu.setEditable(True)
        self.p_nu.addItems(["2048", "4096", "8192", "16384"])
        self.p_nu.setValidator(QIntValidator(16, 65536, self))
        self.p_nu.currentTextChanged.connect(self._changed)
        f.addRow("Pixels across the box", self.p_nu)
        self.p_adaptive = QCheckBox("Adaptive: the largest size that fits, per snapshot")
        self.p_adaptive.setToolTip("Needs 'Check memory': it asks the binary what each snapshot needs at "
                                   "each size (halving down to 1024²) and plans the pipeline accordingly.")
        self.p_adaptive.toggled.connect(self._changed)
        f.addRow("", self.p_adaptive)
        self.p_axis = QComboBox()
        self.p_axis.addItems(["x", "y", "z"])
        self.p_axis.currentIndexChanged.connect(self._changed)
        f.addRow("Normal axis", self.p_axis)
        self.p_center = QLineEdit(placeholderText="box centre")
        self.p_center.editingFinished.connect(self._changed)
        f.addRow("Position (Mpc)", self.p_center)
        self.p_vg = QCheckBox("Velocity gradient at each point")
        self.p_vg.setToolTip("Also evaluate the velocity gradient at every sample point: the divergence, shear and "
                             "vorticity maps (PTS_VEL_GRAD)")
        self.p_vg.toggled.connect(self._changed)
        f.addRow("", self.p_vg)
        self.p_pc = _app._pts_conventions_combo(self._changed)           # the image plane's conventions
        f.addRow(P.label("pts_conventions"), self.p_pc)
        slab = QHBoxLayout()
        self.p_planes = QSpinBox(minimum=1, maximum=256)
        self.p_planes.setToolTip("Sampling planes spread across a slab, which the figures' 'slab' projection averages "
                                 "(the planes option of make_image_plane.py). 1 = one crisp zero-thickness cross-section. "
                                 "A handful of planes ghosts every inclined structure (one displaced copy per plane): "
                                 f"use 1, or at least {rs.PLANE_GHOSTING_MIN} for a smooth projection.")
        self.p_planes.valueChanged.connect(self._changed)
        self.p_thick = QDoubleSpinBox(minimum=0.01, maximum=1000.0, decimals=2, singleStep=0.5)
        self.p_thick.setSuffix(" Mpc")
        self.p_thick.setToolTip("The slab's depth along the normal axis, over which the planes are spread "
                                "(the thickness option of make_image_plane.py; one plane has no thickness)")
        self.p_thick.valueChanged.connect(self._changed)
        slab.addWidget(self.p_planes)
        slab.addWidget(QLabel("planes across"))
        slab.addWidget(self.p_thick)
        slab.addStretch(1)
        f.addRow("Slab", slab)
        sub = QHBoxLayout()
        self.p_super = QSpinBox(minimum=1, maximum=8)
        self.p_super.setToolTip("K: evaluate K×K sample points per pixel and average them (the supersample option of "
                                "make_image_plane.py). The map then estimates the pixel-AREA mean, what the deposit grid "
                                "reports, instead of a point sample: no aliasing where structure is finer than a pixel "
                                "(halo cores, thin caustics). Costs K² points.")
        self.p_super.valueChanged.connect(self._changed)
        self.p_super_note = QLabel()
        sub.addWidget(self.p_super)
        sub.addWidget(self.p_super_note)
        sub.addStretch(1)
        f.addRow("Sub-samples per pixel axis", sub)
        self.p_window = QLineEdit(placeholderText="whole box: u0 u1 v0 v1")
        self.p_window.setToolTip("Image only this part of the plane: horizontal u0..u1 and vertical v0..v1 in Mpc "
                                 "(the u0, u1, v0, v1 options of make_image_plane.py); blank = the whole box. Pixels "
                                 "stay square, so the vertical pixel count follows the window's aspect.")
        self.p_window.editingFinished.connect(self._changed)
        f.addRow("Window (Mpc)", self.p_window)
        col.addWidget(g)

        g, self.p_fields = self._fields_group("Grids")
        self.p_fields_box = g
        gr = QHBoxLayout()
        self.p_grid = _grid_combo(self)
        self.p_grid.currentTextChanged.connect(self._changed)
        gr.addWidget(QLabel("Grid"))
        gr.addWidget(self.p_grid)
        gr.addWidget(QLabel("³ cells    sub-samples per axis"))
        self.p_nsub = QSpinBox(minimum=1, maximum=6)
        self.p_nsub.setToolTip("nSub³ sample points per cell for the sampled deposit")
        self.p_nsub.valueChanged.connect(self._changed)
        gr.addWidget(self.p_nsub)
        gr.addStretch(1)
        g.layout().addLayout(gr, 3, 0, 1, 3)
        dep = QHBoxLayout()
        self.p_deposit = QComboBox()
        self.p_deposit.addItem(P.label("sampled"), "sampled")
        self.p_deposit.addItem(P.label("exact"), "exact")
        self.p_deposit.setItemData(0, P.tooltip("sampled"), Qt.ToolTipRole)
        self.p_deposit.setItemData(1, P.tooltip("exact"), Qt.ToolTipRole)
        self.p_deposit.currentIndexChanged.connect(lambda _i: self._deposit_changed(self.p_deposit, self.p_gpu))
        dep.addWidget(QLabel("Deposit"))
        dep.addWidget(self.p_deposit, 1)
        g.layout().addLayout(dep, 4, 0, 1, 3)
        self.p_gpu = QCheckBox(P.label("gpu"))
        self.p_gpu.setToolTip(P.tooltip("gpu"))
        self.p_gpu.toggled.connect(self._changed)
        g.layout().addWidget(self.p_gpu, 5, 0, 1, 3)
        g.setToolTip("run_ps_pipeline.sh hands these to run_ps_dtfe.sh exactly as a Grids job does; left alone, "
                     "the scripts' own defaults are the production settings (GPU, vertex masses, volume-weighted "
                     "velocities)")
        col.addWidget(g)

        g = QGroupBox("Phase-space options")
        v = QVBoxLayout(g)
        self.p_vm = QCheckBox(P.label("vertex_mass"))
        self.p_vw = QCheckBox(P.label("volume_weighted"))
        self.p_ca = QCheckBox(P.label("caustics"))
        self.p_cu = QCheckBox(P.label("caustic_cusps"))
        self.p_ld = QCheckBox(P.label("linear_deposit"))
        for cb, key in ((self.p_vm, "vertex_mass"), (self.p_vw, "volume_weighted"),
                        (self.p_ca, "caustics"), (self.p_cu, "caustic_cusps"), (self.p_ld, "linear_deposit")):
            cb.setToolTip(P.tooltip(key))
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        _exclusive(self.p_ld, self.p_vw)
        col.addWidget(g)

        g, (self.p_part, self.p_conc, self.p_scratch, self.p_prefix, self.p_pt, self.p_prec,
            self.p_budget, self.p_tess, self.p_om, self.p_so) = self._resources_group()
        col.addWidget(g)

        g = QGroupBox("Run")
        v = QVBoxLayout(g)
        self.p_force = QCheckBox("Recompute every snapshot, up to date or not")
        self.p_plan = QCheckBox("Plan only: show what would run")
        self.p_render = QCheckBox("Then render the point-evaluated figures")
        self.p_force.setToolTip("Ignore every freshness check: recompute even snapshots whose grids are newer than "
                                "their inputs")
        self.p_plan.setToolTip("Print what would run for each snapshot, and compute nothing")
        self.p_render.setToolTip("plot_pointeval.py, on this pipeline's own image plane, with the options below")
        for cb in (self.p_force, self.p_plan, self.p_render):
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        self.p_render_box = QWidget()
        rf = QFormLayout(self.p_render_box)
        rf.setContentsMargins(24, 0, 0, 0)
        self.p_rfields = QLineEdit(placeholderText="all the run wrote: " + ", ".join(rs.POINTEVAL_FIELDS))
        self.p_rfields.setToolTip("A comma-separated subset of the point-evaluated fields (the fields option of "
                                  "plot_pointeval.py); blank = every field the run wrote")
        self.p_rfields.editingFinished.connect(self._changed)
        rf.addRow("Fields", self.p_rfields)
        self.p_froot = QLineEdit(placeholderText=f"default: {rs.FIGURES_ROOT}")
        self.p_froot.setToolTip("Where the rendered figures go")
        self.p_froot.editingFinished.connect(self._changed)
        rf.addRow("Figures folder", _dir_row(self.p_froot, lambda: (self._browse_dir(self.p_froot, "Figures folder"),
                                                                    self._changed())))
        self.p_rproject = QComboBox()
        self.p_rproject.addItem("the central plane: a zero-thickness cross-section", "plane")
        self.p_rproject.addItem("slab: the mean over every plane of the slab", "slab")
        self.p_rproject.setToolTip("The projection option of plot_pointeval.py; 'slab' needs several planes across the "
                                   "slab (above)")
        self.p_rproject.currentIndexChanged.connect(self._changed)
        rf.addRow("Projection", self.p_rproject)
        self.p_rsmooth = QDoubleSpinBox(minimum=0.0, maximum=1000.0, decimals=1, singleStep=0.5)
        self.p_rsmooth.setSuffix(" px")
        self.p_rsmooth.setToolTip("Gaussian smoothing of every map, in output pixels (the smooth option of "
                                  "plot_pointeval.py); 0 = none. Also suppresses the smoothed gradient companions below.")
        self.p_rsmooth.valueChanged.connect(self._changed)
        rf.addRow("Smoothing", self.p_rsmooth)
        self.p_rsmoothd = QDoubleSpinBox(minimum=0.0, maximum=1000.0, decimals=1, singleStep=1.0)
        self.p_rsmoothd.setSuffix(" px")
        self.p_rsmoothd.setToolTip("The gradient maps (divergence, shear, vorticity, density gradient) are piecewise "
                                   "constant per tetrahedron and look faceted; each also gets a Gaussian-smoothed "
                                   "companion figure at this width (the smooth-derivatives option of plot_pointeval.py); "
                                   "0 = none")
        self.p_rsmoothd.valueChanged.connect(self._changed)
        rf.addRow("Gradient maps also smoothed at", self.p_rsmoothd)
        self.p_rfixed = QCheckBox("Fixed density range of the grid maps (0.1 .. 1e4)")
        self.p_rfixed.setToolTip("The fixed-range option of plot_pointeval.py: side-by-side comparison with the grid "
                                 "slice maps; clips ~26% of a z=0 image to the darkest colour")
        self.p_rforce = QCheckBox("Re-render figures newer than their data")
        self.p_rforce.setToolTip("The force option of plot_pointeval.py")
        for cb in (self.p_rfixed, self.p_rforce):
            cb.toggled.connect(self._changed)
            rf.addRow("", cb)
        v.addWidget(self.p_render_box)
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    # ================================================================ Plots tab
    def _build_plots_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Figure sets")
        v = QVBoxLayout(g)
        self.fig_list = QListWidget()
        self.fig_list.setMaximumHeight(200)
        for fs in rs.FIGURE_SETS:
            it = QListWidgetItem(fs.title)
            it.setData(Qt.UserRole, fs.key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.fig_list.addItem(it)
        self.fig_list.itemChanged.connect(self._changed)
        v.addWidget(self.fig_list)
        self.fig_stack = QStackedWidget()
        for fs in rs.FIGURE_SETS:
            self.fig_stack.addWidget(self._figure_page(fs))
        self.fig_list.currentRowChanged.connect(self.fig_stack.setCurrentIndex)
        v.addWidget(self.fig_stack)
        col.addWidget(g)

        g = QGroupBox("Grids to plot")
        f = QFormLayout(g)
        self.pl_method = QComboBox()
        for key, label in (("auto", "Auto: PS-DTFE where present, else DTFE"), ("ps", "PS-DTFE"),
                           ("dtfe", "Standard DTFE")):
            self.pl_method.addItem(label, key)
        self.pl_method.currentIndexChanged.connect(self._changed)
        f.addRow("Estimator", self.pl_method)
        self.pl_prefix = QLineEdit(placeholderText="default (ps_output)")
        self.pl_prefix.editingFinished.connect(self._changed)
        f.addRow("Grid prefix", self.pl_prefix)
        self.pl_smooth = QDoubleSpinBox(minimum=0.0, maximum=50.0, decimals=1, singleStep=0.5)
        self.pl_smooth.setSuffix(" cells")
        self.pl_smooth.valueChanged.connect(self._changed)
        f.addRow("Smoothing", self.pl_smooth)
        self.pl_froot = QLineEdit(placeholderText=f"default: {rs.FIGURES_ROOT}")
        self.pl_froot.setToolTip(f"Where the point-evaluated maps go (the other figure sets write under {rs.FIGURES_ROOT})")
        self.pl_froot.editingFinished.connect(self._changed)
        f.addRow("Figures folder", _dir_row(self.pl_froot, lambda: (self._browse_dir(self.pl_froot, "Figures folder"),
                                                                     self._changed())))
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _figure_page(self, fs: rs.FigureSet) -> QWidget:
        page = QWidget()
        f = QFormLayout(page)
        f.setContentsMargins(4, 4, 4, 4)
        about = QLabel(fs.about)
        about.setWordWrap(True)
        about.setToolTip(f"python/{fs.script}, {_app.PER_TEXT[fs.per]}  →  {rs.FIGURES_ROOT / fs.output}")
        f.addRow(about)
        rows: dict[str, QHBoxLayout] = {}
        for o in fs.opts:
            if o.kind == "bool":
                w = QCheckBox(o.label)
                w.toggled.connect(self._changed)
            elif o.kind == "int":
                w = QSpinBox(minimum=0, maximum=1_000_000)
                w.valueChanged.connect(self._changed)
            elif o.kind == "float":
                lo, hi, dec = o.limits or (0.0, 1000.0, 2)
                w = QDoubleSpinBox(decimals=dec, singleStep=0.5)     # decimals FIRST: the range rounds to them
                w.setRange(lo, hi)
                w.valueChanged.connect(self._changed)
            elif o.kind == "choice":
                w = QComboBox()
                w.addItems(list(o.choices))
                w.currentIndexChanged.connect(self._changed)
            elif o.kind == "plane":
                w = QComboBox()
                w.currentIndexChanged.connect(self._changed)
            elif o.kind == "multi":
                w = QListWidget()
                for choice in o.choices:
                    it = QListWidgetItem(choice)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Unchecked)
                    w.addItem(it)
                w.setMaximumHeight(min(len(o.choices), 8) * 20 + 8)
                w.itemChanged.connect(self._changed)
                _checkable(w)
            else:
                w = QLineEdit(placeholderText=o.help)
                w.editingFinished.connect(self._changed)
            if o.help:
                w.setToolTip(o.help)
            self.opt_widgets[(fs.key, o.key)] = (o, w)
            if not o.row:
                f.addRow("" if o.kind == "bool" else o.label, w)
                continue
            # options sharing a row title sit side by side under it, each with its short label
            if o.row not in rows:
                line = QHBoxLayout()
                line.setContentsMargins(0, 0, 0, 0)
                rows[o.row] = line
                title = QLabel(o.row)
                f.addRow(title, line)
            line = rows[o.row]
            if o.kind != "bool":
                lab = QLabel(o.label)
                lab.setToolTip(o.help)
                line.addWidget(lab)
            if o.kind == "text":
                w.setPlaceholderText("")
                w.setMaximumWidth(90)
            line.addWidget(w, 1 if o.kind == "text" else 0)
        for line in rows.values():
            line.addStretch(1)
        return page


import app as _app  # noqa: E402
