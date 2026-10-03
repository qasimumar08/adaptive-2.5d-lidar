// =============================================================================
// grid_bindings.cpp — pybind11 Python Bindings for Foveated Grid Engine
// PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cell.h"
#include "zone_manager.h"
#include "foveated_grid.h"
#include "hip_projection.h"

#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>

namespace py = pybind11;
using namespace foveated_grid;

namespace {

/// Helper to convert a NumPy array into std::vector<ClassifiedPoint>.
/// Supports arrays of shape (N, 3), (N, 4), or (N, 5):
///   - (N, 3): [x, y, z] -> semantic_class = 5, confidence = 1.0
///   - (N, 4): [x, y, z, class] -> confidence = 1.0
///   - (N, 5): [x, y, z, class, confidence]
std::vector<ClassifiedPoint> parsePointsArray(
    py::array_t<float, py::array::c_style | py::array::forcecast> arr
) {
    py::buffer_info buf = arr.request();

    if (buf.ndim == 1 && buf.shape[0] == 0) {
        return {};
    }

    if (buf.ndim != 2) {
        throw std::invalid_argument(
            "Points array must be 2D with shape (N, 3), (N, 4), or (N, 5), but got ndim = " +
            std::to_string(buf.ndim)
        );
    }

    size_t num_points = static_cast<size_t>(buf.shape[0]);
    size_t num_cols   = static_cast<size_t>(buf.shape[1]);

    if (num_cols < 3) {
        throw std::invalid_argument(
            "Points array must have at least 3 columns (x, y, z), but got " +
            std::to_string(num_cols)
        );
    }

    std::vector<ClassifiedPoint> points;
    points.reserve(num_points);

    const float* ptr = static_cast<const float*>(buf.ptr);

    for (size_t i = 0; i < num_points; ++i) {
        const float* row = ptr + i * num_cols;
        ClassifiedPoint p;
        p.x = row[0];
        p.y = row[1];
        p.z = row[2];
        p.semantic_class = (num_cols >= 4) ? static_cast<uint8_t>(row[3]) : static_cast<uint8_t>(5);
        p.confidence     = (num_cols >= 5) ? row[4] : 1.0f;
        points.push_back(p);
    }

    return points;
}

} // anonymous namespace

PYBIND11_MODULE(grid_py, m) {
    m.doc() = "Python bindings for the Adaptive Variable Resolution 2.5D Foveated Grid Engine (AMD ROCm / HIP)";

    // -------------------------------------------------------------------------
    // Cell Struct
    // -------------------------------------------------------------------------
    py::class_<Cell>(m, "Cell", "2.5D elevation grid cell storing elevation, class, and occupancy stats")
        .def(py::init<>())
        .def_readonly("elevation_min", &Cell::elevation_min, "Minimum observed elevation in meters")
        .def_readonly("elevation_max", &Cell::elevation_max, "Maximum observed elevation in meters")
        .def_readonly("elevation_sum", &Cell::elevation_sum, "Sum of elevations in meters for mean computation")
        .def_readonly("semantic_class", &Cell::semantic_class, "Dominant semantic class (0-5)")
        .def_readonly("confidence", &Cell::confidence, "Mean segmentation confidence [0, 1]")
        .def_readonly("occupancy", &Cell::occupancy, "Occupancy state (0=free, 1=occupied, 2=unknown)")
        .def_readonly("point_count", &Cell::point_count, "Number of points projected into this cell")
        .def("elevation_mean", &Cell::elevation_mean, "Compute mean elevation of the cell")
        .def("height_span", &Cell::height_span, "Compute height span (elevation_max - elevation_min)")
        .def("reset", &Cell::reset, "Reset cell to unobserved state")
        .def("__repr__", [](const Cell& c) {
            return "<Cell mean_z=" + std::to_string(c.elevation_mean()) +
                   " class=" + std::to_string(static_cast<int>(c.semantic_class)) +
                   " occ=" + std::to_string(static_cast<int>(c.occupancy)) +
                   " pts=" + std::to_string(c.point_count) + ">";
        });

    // -------------------------------------------------------------------------
    // ZoneConfig Struct
    // -------------------------------------------------------------------------
    py::class_<ZoneConfig>(m, "ZoneConfig", "Configuration properties for a resolution zone")
        .def_readonly("name", &ZoneConfig::name, "Zone name (e.g. immediate, near, mid, far)")
        .def_readonly("r_min", &ZoneConfig::r_min, "Inner radius in meters")
        .def_readonly("r_max", &ZoneConfig::r_max, "Outer radius in meters")
        .def_readonly("cell_size", &ZoneConfig::cell_size, "Grid cell size in meters")
        .def_readonly("grid_width", &ZoneConfig::grid_width, "Width in cells")
        .def_readonly("grid_height", &ZoneConfig::grid_height, "Height in cells")
        .def_readonly("offset_x", &ZoneConfig::offset_x, "Center offset along X")
        .def_readonly("offset_y", &ZoneConfig::offset_y, "Center offset along Y")
        .def_readonly("base_idx", &ZoneConfig::base_idx, "Base cell index in flat array")
        .def_readonly("total_cells", &ZoneConfig::total_cells, "Total cells in this zone")
        .def("__repr__", [](const ZoneConfig& z) {
            return "<ZoneConfig " + z.name + " [" + std::to_string(z.r_min) +
                   "m - " + std::to_string(z.r_max) + "m] res=" +
                   std::to_string(z.cell_size) + "m (" +
                   std::to_string(z.grid_width) + "x" + std::to_string(z.grid_height) + ")>";
        });

    // -------------------------------------------------------------------------
    // ClassifiedPoint Struct
    // -------------------------------------------------------------------------
    py::class_<ClassifiedPoint>(m, "ClassifiedPoint", "Point with 3D coordinates, semantic class, and confidence")
        .def(py::init<float, float, float, uint8_t, float>(),
             py::arg("x") = 0.0f,
             py::arg("y") = 0.0f,
             py::arg("z") = 0.0f,
             py::arg("semantic_class") = 5,
             py::arg("confidence") = 1.0f)
        .def_readwrite("x", &ClassifiedPoint::x)
        .def_readwrite("y", &ClassifiedPoint::y)
        .def_readwrite("z", &ClassifiedPoint::z)
        .def_readwrite("semantic_class", &ClassifiedPoint::semantic_class)
        .def_readwrite("confidence", &ClassifiedPoint::confidence)
        .def("__repr__", [](const ClassifiedPoint& p) {
            return "<ClassifiedPoint (" + std::to_string(p.x) + ", " +
                   std::to_string(p.y) + ", " + std::to_string(p.z) +
                   ") class=" + std::to_string(static_cast<int>(p.semantic_class)) +
                   " conf=" + std::to_string(p.confidence) + ">";
        });

    // -------------------------------------------------------------------------
    // ZoneManager Class
    // -------------------------------------------------------------------------
    py::class_<ZoneManager>(m, "ZoneManager", "Manages concentric multi-resolution grid zones")
        .def(py::init<>())
        .def("addZone", &ZoneManager::addZone,
             py::arg("name"), py::arg("r_min"), py::arg("r_max"), py::arg("cell_size"),
             "Add a concentric zone (must be in order of increasing radii)")
        .def("add_zone", &ZoneManager::addZone,
             py::arg("name"), py::arg("r_min"), py::arg("r_max"), py::arg("cell_size"))
        .def("finalize", &ZoneManager::finalize, "Compute zone grid dimensions and offsets")
        .def("getZoneIndex", &ZoneManager::getZoneIndex, py::arg("radial_distance"),
             "Lookup zone index by radial distance (returns -1 if out of bounds)")
        .def("get_zone_index", &ZoneManager::getZoneIndex, py::arg("radial_distance"))
        .def("getCellIndex", &ZoneManager::getCellIndex,
             py::arg("zone_idx"), py::arg("x"), py::arg("y"),
             "Compute flat cell index for (x, y) coordinates in the specified zone")
        .def("get_cell_index", &ZoneManager::getCellIndex,
             py::arg("zone_idx"), py::arg("x"), py::arg("y"))
        .def("totalCells", &ZoneManager::totalCells, "Total cells across all zones")
        .def("total_cells", &ZoneManager::totalCells)
        .def("numZones", &ZoneManager::numZones, "Number of configured zones")
        .def("num_zones", &ZoneManager::numZones)
        .def("zones", &ZoneManager::zones, py::return_value_policy::reference_internal);

    // -------------------------------------------------------------------------
    // FoveatedGrid Class
    // -------------------------------------------------------------------------
    py::class_<FoveatedGrid>(m, "FoveatedGrid", "Multi-resolution 2.5D foveated elevation & semantic grid")
        .def(py::init<const ZoneManager&>(), py::arg("zone_manager"))
        .def("clear", &FoveatedGrid::clear, "Reset all cells to default unobserved state")

        // Overloaded projectPoints: std::vector<ClassifiedPoint> or 2D NumPy array
        .def("projectPoints", [](FoveatedGrid& self, const std::vector<ClassifiedPoint>& points) {
            self.projectPoints(points);
        }, py::arg("points"), "Project points using CPU accumulation (list of ClassifiedPoint)")
        .def("projectPoints", [](FoveatedGrid& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            std::vector<ClassifiedPoint> pts = parsePointsArray(arr);
            self.projectPoints(pts);
        }, py::arg("points"), "Project points using CPU accumulation (NumPy 2D array shape (N, 3..5))")
        .def("project_points", [](FoveatedGrid& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            std::vector<ClassifiedPoint> pts = parsePointsArray(arr);
            self.projectPoints(pts);
        }, py::arg("points"))

        // Overloaded projectPointsGPU: std::vector<ClassifiedPoint> or 2D NumPy array
        .def("projectPointsGPU", [](FoveatedGrid& self, const std::vector<ClassifiedPoint>& points) {
            self.projectPointsGPU(points);
        }, py::arg("points"), "Project points using AMD ROCm HIP GPU kernel (list of ClassifiedPoint)")
        .def("projectPointsGPU", [](FoveatedGrid& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            std::vector<ClassifiedPoint> pts = parsePointsArray(arr);
            self.projectPointsGPU(pts);
        }, py::arg("points"), "Project points using AMD ROCm HIP GPU kernel (NumPy 2D array shape (N, 3..5))")
        .def("project_points_gpu", [](FoveatedGrid& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            std::vector<ClassifiedPoint> pts = parsePointsArray(arr);
            self.projectPointsGPU(pts);
        }, py::arg("points"))

        .def("finalizeCells", &FoveatedGrid::finalizeCells,
             py::arg("unknown_threshold") = 0, py::arg("confidence_threshold") = 0.5f,
             "Compute mean elevation, majority vote semantic classes, and set occupancy")
        .def("finalize_cells", &FoveatedGrid::finalizeCells,
             py::arg("unknown_threshold") = 0, py::arg("confidence_threshold") = 0.5f)

        .def("getCell", &FoveatedGrid::getCell,
             py::arg("zone_idx"), py::arg("grid_x"), py::arg("grid_y"),
             py::return_value_policy::reference_internal,
             "Access cell at zone_idx and 2D grid coordinates (grid_x, grid_y)")
        .def("get_cell", &FoveatedGrid::getCell,
             py::arg("zone_idx"), py::arg("grid_x"), py::arg("grid_y"),
             py::return_value_policy::reference_internal)

        .def("cells", &FoveatedGrid::cells, py::return_value_policy::reference_internal,
             "Flat list of all grid cells across all zones")
        .def("totalCells", &FoveatedGrid::totalCells, "Total cells in the grid")
        .def("total_cells", &FoveatedGrid::totalCells)
        .def("memoryUsage", &FoveatedGrid::memoryUsage, "Total cell memory in bytes")
        .def("memory_usage", &FoveatedGrid::memoryUsage)
        .def("zoneManager", &FoveatedGrid::zoneManager, py::return_value_policy::reference_internal)
        .def("zone_manager", &FoveatedGrid::zoneManager, py::return_value_policy::reference_internal)

        // Fast NumPy map extractors for visualization and dashboard
        .def("get_elevation_map", [](const FoveatedGrid& self, int zone_idx) {
            const auto& zm = self.zoneManager();
            if (zone_idx < 0 || zone_idx >= zm.numZones()) {
                throw std::out_of_range("Zone index out of range: " + std::to_string(zone_idx));
            }
            const auto& z = zm.zones()[zone_idx];
            py::array_t<float> result({z.grid_height, z.grid_width});
            py::buffer_info buf = result.request();
            float* ptr = static_cast<float*>(buf.ptr);

            const auto& all_cells = self.cells();
            const Cell* cell_ptr = &all_cells[z.base_idx];
            const int total = z.grid_height * z.grid_width;
            for (int i = 0; i < total; ++i) {
                ptr[i] = (cell_ptr[i].point_count > 0) ? cell_ptr[i].elevation_mean() : NAN;
            }
            return result;
        }, py::arg("zone_idx"), "Return 2D numpy float array of mean elevation for a zone (NaN if unobserved)")

        .def("get_semantic_map", [](const FoveatedGrid& self, int zone_idx) {
            const auto& zm = self.zoneManager();
            if (zone_idx < 0 || zone_idx >= zm.numZones()) {
                throw std::out_of_range("Zone index out of range: " + std::to_string(zone_idx));
            }
            const auto& z = zm.zones()[zone_idx];
            py::array_t<uint8_t> result({z.grid_height, z.grid_width});
            py::buffer_info buf = result.request();
            uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

            const auto& all_cells = self.cells();
            const Cell* cell_ptr = &all_cells[z.base_idx];
            const int total = z.grid_height * z.grid_width;
            for (int i = 0; i < total; ++i) {
                ptr[i] = cell_ptr[i].semantic_class;
            }
            return result;
        }, py::arg("zone_idx"), "Return 2D numpy uint8 array of semantic classes for a zone")

        .def("get_occupancy_map", [](const FoveatedGrid& self, int zone_idx) {
            const auto& zm = self.zoneManager();
            if (zone_idx < 0 || zone_idx >= zm.numZones()) {
                throw std::out_of_range("Zone index out of range: " + std::to_string(zone_idx));
            }
            const auto& z = zm.zones()[zone_idx];
            py::array_t<uint8_t> result({z.grid_height, z.grid_width});
            py::buffer_info buf = result.request();
            uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

            const auto& all_cells = self.cells();
            const Cell* cell_ptr = &all_cells[z.base_idx];
            const int total = z.grid_height * z.grid_width;
            for (int i = 0; i < total; ++i) {
                ptr[i] = cell_ptr[i].occupancy;
            }
            return result;
        }, py::arg("zone_idx"), "Return 2D numpy uint8 array of occupancy states (0=free, 1=occupied, 2=unknown) for a zone")

        .def("get_point_count_map", [](const FoveatedGrid& self, int zone_idx) {
            const auto& zm = self.zoneManager();
            if (zone_idx < 0 || zone_idx >= zm.numZones()) {
                throw std::out_of_range("Zone index out of range: " + std::to_string(zone_idx));
            }
            const auto& z = zm.zones()[zone_idx];
            py::array_t<uint16_t> result({z.grid_height, z.grid_width});
            py::buffer_info buf = result.request();
            uint16_t* ptr = static_cast<uint16_t*>(buf.ptr);

            const auto& all_cells = self.cells();
            const Cell* cell_ptr = &all_cells[z.base_idx];
            const int total = z.grid_height * z.grid_width;
            for (int i = 0; i < total; ++i) {
                ptr[i] = cell_ptr[i].point_count;
            }
            return result;
        }, py::arg("zone_idx"), "Return 2D numpy uint16 array of point counts for a zone");

    // -------------------------------------------------------------------------
    // Utility functions
    // -------------------------------------------------------------------------
    m.def("is_hip_available", &isHipAvailable, "Check if AMD ROCm HIP GPU device is active and available");
}
