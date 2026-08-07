#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/PLY.h>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/PLY.h>
#include <CGAL/boost/graph/border.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/version.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace PMP = CGAL::Polygon_mesh_processing;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Plane = Kernel::Plane_3;
using Vector = Kernel::Vector_3;
using Mesh = CGAL::Surface_mesh<Point>;
using Json = nlohmann::json;

namespace {

constexpr int kSchemaVersion = 1;
constexpr const char *kAlgorithmVersion = "1.0.0";

class Sha256 {
public:
  void update(const std::uint8_t *data, std::size_t length) {
    for (std::size_t i = 0; i < length; ++i) {
      block_[block_size_++] = data[i];
      bit_length_ += 8;
      if (block_size_ == block_.size()) {
        transform();
        block_size_ = 0;
      }
    }
  }

  void update(const std::string &value) {
    update(reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
  }

  std::string finish() {
    const std::uint64_t original_bit_length = bit_length_;
    block_[block_size_++] = 0x80;
    if (block_size_ > 56) {
      while (block_size_ < 64)
        block_[block_size_++] = 0;
      transform();
      block_size_ = 0;
    }
    while (block_size_ < 56)
      block_[block_size_++] = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[block_size_++] =
          static_cast<std::uint8_t>(original_bit_length >> shift);
    }
    transform();

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::uint32_t word : state_)
      output << std::setw(8) << word;
    return output.str();
  }

private:
  static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  static std::uint32_t rotate_right(std::uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
  }

  void transform() {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t offset = i * 4;
      words[i] = (static_cast<std::uint32_t>(block_[offset]) << 24) |
                 (static_cast<std::uint32_t>(block_[offset + 1]) << 16) |
                 (static_cast<std::uint32_t>(block_[offset + 2]) << 8) |
                 static_cast<std::uint32_t>(block_[offset + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^
                               rotate_right(words[i - 15], 18) ^
                               (words[i - 15] >> 3);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^
                               rotate_right(words[i - 2], 19) ^
                               (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const std::uint32_t sum1 =
          rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temp1 =
          h + sum1 + choice + kRoundConstants[i] + words[i];
      const std::uint32_t sum0 =
          rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                         0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                         0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t bit_length_ = 0;
};

std::string sha256_text(const std::string &value) {
  Sha256 digest;
  digest.update(value);
  return digest.finish();
}

std::string sha256_file(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open input mesh: " + path.string());
  Sha256 digest;
  std::array<char, 64 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      digest.update(reinterpret_cast<const std::uint8_t *>(buffer.data()),
                    static_cast<std::size_t>(count));
    }
  }
  if (!input.eof())
    throw std::runtime_error("failed while reading input mesh: " +
                             path.string());
  return digest.finish();
}

double coordinate(double value) { return value == 0.0 ? 0.0 : value; }

using PointKey = std::tuple<double, double, double>;
using Edge = std::pair<std::size_t, std::size_t>;
using Face = std::array<std::size_t, 3>;

PointKey point_key(const Point &point) {
  return {coordinate(CGAL::to_double(point.x())),
          coordinate(CGAL::to_double(point.y())),
          coordinate(CGAL::to_double(point.z()))};
}

Edge edge_key(std::size_t a, std::size_t b) {
  return a < b ? Edge{a, b} : Edge{b, a};
}

Face face_key(Face face) {
  std::sort(face.begin(), face.end());
  return face;
}

std::array<double, 3> point_array(const Point &point) {
  return {coordinate(CGAL::to_double(point.x())),
          coordinate(CGAL::to_double(point.y())),
          coordinate(CGAL::to_double(point.z()))};
}

std::string stable_point_text(const Point &point) {
  const auto values = point_array(point);
  std::ostringstream output;
  output << std::hexfloat << values[0] << ',' << values[1] << ',' << values[2];
  return output.str();
}

double vector_length(const Vector &value) {
  return std::sqrt(std::max(0.0, CGAL::to_double(value.squared_length())));
}

double triangle_area(const Point &a, const Point &b, const Point &c) {
  return 0.5 * vector_length(CGAL::cross_product(b - a, c - a));
}

Json bounding_box_json(const std::vector<Point> &points,
                       const std::set<std::size_t> *selected = nullptr) {
  std::array<double, 3> minimum = {std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity()};
  std::array<double, 3> maximum = {-std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity()};
  auto include = [&](std::size_t index) {
    const auto value = point_array(points[index]);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      minimum[axis] = std::min(minimum[axis], value[axis]);
      maximum[axis] = std::max(maximum[axis], value[axis]);
    }
  };
  if (selected) {
    for (const auto index : *selected)
      include(index);
  } else {
    for (std::size_t index = 0; index < points.size(); ++index)
      include(index);
  }
  return {{"min", minimum}, {"max", maximum}};
}

struct DisjointSet {
  explicit DisjointSet(std::size_t size) : parent(size), rank(size, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  std::size_t find(std::size_t value) {
    if (parent[value] != value)
      parent[value] = find(parent[value]);
    return parent[value];
  }

  void unite(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return;
    if (rank[a] < rank[b])
      std::swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b])
      ++rank[a];
  }

  std::vector<std::size_t> parent;
  std::vector<unsigned char> rank;
};

struct Component {
  std::string id;
  std::vector<std::size_t> faces;
  std::set<std::size_t> vertices;
  double area = 0.0;
};

struct BoundaryLoop {
  std::vector<std::size_t> vertices;
  std::size_t face = 0;
};

std::vector<std::size_t> normalize_loop(const std::vector<std::size_t> &loop,
                                        const std::vector<Point> &points) {
  if (loop.empty())
    return {};
  std::vector<std::size_t> best;
  std::vector<std::size_t> candidate;
  candidate.reserve(loop.size());
  auto candidate_text = [&](const std::vector<std::size_t> &values) {
    std::string text;
    for (const auto index : values) {
      text += stable_point_text(points[index]);
      text.push_back(';');
    }
    return text;
  };
  std::optional<std::string> best_text;
  for (bool reverse : {false, true}) {
    for (std::size_t start = 0; start < loop.size(); ++start) {
      candidate.clear();
      for (std::size_t offset = 0; offset < loop.size(); ++offset) {
        const std::size_t position =
            reverse ? (start + loop.size() - offset) % loop.size()
                    : (start + offset) % loop.size();
        candidate.push_back(loop[position]);
      }
      const std::string text = candidate_text(candidate);
      if (!best_text || text < *best_text) {
        best_text = text;
        best = candidate;
      }
    }
  }
  return best;
}

std::string canonical_loop_identity(const std::vector<Point> &points) {
  if (points.empty())
    return {};
  std::optional<std::string> best;
  for (bool reverse : {false, true}) {
    for (std::size_t start = 0; start < points.size(); ++start) {
      std::string candidate;
      for (std::size_t offset = 0; offset < points.size(); ++offset) {
        const std::size_t position =
            reverse ? (start + points.size() - offset) % points.size()
                    : (start + offset) % points.size();
        candidate += stable_point_text(points[position]);
        candidate.push_back(';');
      }
      if (!best || candidate < *best)
        best = std::move(candidate);
    }
  }
  return *best;
}

std::string loop_id_from_points(const std::vector<Point> &points) {
  return "loop-" + sha256_text(canonical_loop_identity(points)).substr(0, 20);
}

std::vector<BoundaryLoop>
find_boundary_loops(const std::map<Edge, std::vector<std::size_t>> &edge_faces,
                    const std::vector<Point> &points,
                    std::size_t &non_cycle_boundary_component_count) {
  std::map<std::size_t, std::vector<std::size_t>> adjacency;
  std::map<Edge, std::size_t> boundary_face;
  for (const auto &[edge, faces] : edge_faces) {
    if (faces.size() != 1)
      continue;
    adjacency[edge.first].push_back(edge.second);
    adjacency[edge.second].push_back(edge.first);
    boundary_face[edge] = faces.front();
  }
  for (auto &[vertex, neighbors] : adjacency) {
    (void)vertex;
    std::sort(neighbors.begin(), neighbors.end());
  }

  std::set<Edge> visited;
  std::vector<BoundaryLoop> loops;
  non_cycle_boundary_component_count = 0;
  for (const auto &[initial_edge, initial_face] : boundary_face) {
    if (visited.contains(initial_edge))
      continue;
    std::set<std::size_t> component_vertices;
    std::queue<std::size_t> pending;
    pending.push(initial_edge.first);
    component_vertices.insert(initial_edge.first);
    while (!pending.empty()) {
      const auto vertex = pending.front();
      pending.pop();
      for (const auto neighbor : adjacency[vertex]) {
        visited.insert(edge_key(vertex, neighbor));
        if (component_vertices.insert(neighbor).second)
          pending.push(neighbor);
      }
    }
    const bool is_cycle = std::all_of(
        component_vertices.begin(), component_vertices.end(),
        [&](std::size_t vertex) { return adjacency[vertex].size() == 2; });
    if (!is_cycle) {
      ++non_cycle_boundary_component_count;
      continue;
    }

    const auto start =
        *std::min_element(component_vertices.begin(), component_vertices.end(),
                          [&](std::size_t a, std::size_t b) {
                            return point_key(points[a]) < point_key(points[b]);
                          });
    std::vector<std::size_t> loop;
    std::size_t previous = std::numeric_limits<std::size_t>::max();
    std::size_t current = start;
    do {
      loop.push_back(current);
      const auto &neighbors = adjacency[current];
      std::size_t next = neighbors.front();
      if (next == previous)
        next = neighbors.back();
      previous = current;
      current = next;
      if (loop.size() > component_vertices.size()) {
        loop.clear();
        break;
      }
    } while (current != start);
    if (loop.size() != component_vertices.size()) {
      ++non_cycle_boundary_component_count;
      continue;
    }
    loop = normalize_loop(loop, points);
    const std::size_t face = boundary_face.at(edge_key(loop[0], loop[1]));
    loops.push_back({std::move(loop), face});
    (void)initial_face;
  }
  std::sort(loops.begin(), loops.end(),
            [&](const BoundaryLoop &a, const BoundaryLoop &b) {
              std::string a_text;
              std::string b_text;
              for (auto vertex : a.vertices)
                a_text += stable_point_text(points[vertex]);
              for (auto vertex : b.vertices)
                b_text += stable_point_text(points[vertex]);
              return a_text < b_text;
            });
  return loops;
}

std::array<double, 3> normalized_plane_normal(const Plane &plane) {
  Vector vector = plane.orthogonal_vector();
  const double length = vector_length(vector);
  if (!(length > 0.0))
    return {0.0, 0.0, 1.0};
  std::array<double, 3> normal = {CGAL::to_double(vector.x()) / length,
                                  CGAL::to_double(vector.y()) / length,
                                  CGAL::to_double(vector.z()) / length};
  std::size_t dominant = 0;
  for (std::size_t axis = 1; axis < 3; ++axis) {
    if (std::abs(normal[axis]) > std::abs(normal[dominant]))
      dominant = axis;
  }
  if (normal[dominant] < 0.0) {
    for (double &value : normal)
      value = -value;
  }
  return normal;
}

Json loop_json(const BoundaryLoop &loop, const std::vector<Point> &points,
               const std::vector<Face> &faces,
               const std::map<Edge, std::vector<std::size_t>> &edge_faces,
               const std::string &component_id, const Json &mesh_bounds) {
  std::vector<Point> loop_points;
  loop_points.reserve(loop.vertices.size());
  for (const auto index : loop.vertices) {
    loop_points.push_back(points[index]);
  }

  Plane plane;
  CGAL::linear_least_squares_fitting_3(loop_points.begin(), loop_points.end(),
                                       plane, CGAL::Dimension_tag<0>());
  const auto normal = normalized_plane_normal(plane);
  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  for (const Point &point : loop_points) {
    cx += CGAL::to_double(point.x());
    cy += CGAL::to_double(point.y());
    cz += CGAL::to_double(point.z());
  }
  const double inverse_count = 1.0 / static_cast<double>(loop_points.size());
  const Point centroid(cx * inverse_count, cy * inverse_count,
                       cz * inverse_count);
  const Point origin = plane.projection(centroid);

  std::array<double, 3> reference = {1.0, 0.0, 0.0};
  if (std::abs(normal[1]) < std::abs(normal[0]) &&
      std::abs(normal[1]) <= std::abs(normal[2])) {
    reference = {0.0, 1.0, 0.0};
  } else if (std::abs(normal[2]) < std::abs(normal[0]) &&
             std::abs(normal[2]) < std::abs(normal[1])) {
    reference = {0.0, 0.0, 1.0};
  }
  std::array<double, 3> axis_u = {
      normal[1] * reference[2] - normal[2] * reference[1],
      normal[2] * reference[0] - normal[0] * reference[2],
      normal[0] * reference[1] - normal[1] * reference[0]};
  const double axis_u_length = std::sqrt(
      axis_u[0] * axis_u[0] + axis_u[1] * axis_u[1] + axis_u[2] * axis_u[2]);
  for (double &value : axis_u)
    value /= axis_u_length;
  const std::array<double, 3> axis_v = {
      normal[1] * axis_u[2] - normal[2] * axis_u[1],
      normal[2] * axis_u[0] - normal[0] * axis_u[2],
      normal[0] * axis_u[1] - normal[1] * axis_u[0]};

  double perimeter = 0.0;
  double twice_area = 0.0;
  double diameter = 0.0;
  double squared_residual_sum = 0.0;
  std::vector<std::array<double, 2>> projected;
  projected.reserve(loop_points.size());
  const auto origin_values = point_array(origin);
  for (std::size_t i = 0; i < loop_points.size(); ++i) {
    const Point &point = loop_points[i];
    const Point &next = loop_points[(i + 1) % loop_points.size()];
    perimeter +=
        std::sqrt(CGAL::to_double(CGAL::squared_distance(point, next)));
    squared_residual_sum +=
        CGAL::to_double(CGAL::squared_distance(point, plane));
    const auto value = point_array(point);
    const std::array<double, 3> relative = {value[0] - origin_values[0],
                                            value[1] - origin_values[1],
                                            value[2] - origin_values[2]};
    projected.push_back({relative[0] * axis_u[0] + relative[1] * axis_u[1] +
                             relative[2] * axis_u[2],
                         relative[0] * axis_v[0] + relative[1] * axis_v[1] +
                             relative[2] * axis_v[2]});
    for (std::size_t j = i + 1; j < loop_points.size(); ++j) {
      diameter = std::max(diameter,
                          std::sqrt(CGAL::to_double(
                              CGAL::squared_distance(point, loop_points[j]))));
    }
  }
  for (std::size_t i = 0; i < projected.size(); ++i) {
    const auto &a = projected[i];
    const auto &b = projected[(i + 1) % projected.size()];
    twice_area += a[0] * b[1] - b[0] * a[1];
  }

  Vector normal_sum(0.0, 0.0, 0.0);
  std::size_t normal_count = 0;
  for (std::size_t i = 0; i < loop.vertices.size(); ++i) {
    const Edge edge = edge_key(loop.vertices[i],
                               loop.vertices[(i + 1) % loop.vertices.size()]);
    const auto found = edge_faces.find(edge);
    if (found == edge_faces.end() || found->second.size() != 1)
      continue;
    const Face &face = faces[found->second.front()];
    Vector face_normal = CGAL::cross_product(points[face[1]] - points[face[0]],
                                             points[face[2]] - points[face[0]]);
    const double length = vector_length(face_normal);
    if (length > 0.0) {
      normal_sum = normal_sum + face_normal / length;
      ++normal_count;
    }
  }
  const double normal_coherence =
      normal_count == 0
          ? 0.0
          : std::clamp(vector_length(normal_sum) / normal_count, 0.0, 1.0);

  const auto box_min = mesh_bounds.at("min").get<std::array<double, 3>>();
  const auto box_max = mesh_bounds.at("max").get<std::array<double, 3>>();
  double bbox_distance = std::numeric_limits<double>::infinity();
  for (const auto &point : loop_points) {
    const auto value = point_array(point);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      bbox_distance = std::min(bbox_distance, value[axis] - box_min[axis]);
      bbox_distance = std::min(bbox_distance, box_max[axis] - value[axis]);
    }
  }
  bbox_distance = std::max(0.0, bbox_distance);

  Json ordered_positions = Json::array();
  for (const auto &point : loop_points)
    ordered_positions.push_back(point_array(point));
  return {
      {"loopId", loop_id_from_points(loop_points)},
      {"componentId", component_id},
      {"orderedBoundaryPositions", std::move(ordered_positions)},
      {"perimeterM", perimeter},
      {"approximateEnclosedAreaM2", std::abs(twice_area) * 0.5},
      {"diameterM", diameter},
      {"bestFitPlane", {{"origin", point_array(origin)}, {"normal", normal}}},
      {"planeRmsResidualM",
       std::sqrt(std::max(0.0, squared_residual_sum * inverse_count))},
      {"boundaryNormalCoherence", normal_coherence},
      {"distanceFromMeshBoundingBoxBoundaryM", bbox_distance},
      {"vertexCount", loop.vertices.size()}};
}

struct ParsedArguments {
  std::string command;
  fs::path input;
  fs::path policy;
  fs::path output;
  fs::path report;
  double voxel_size_m = 0.0;
  bool json_version = false;
};

ParsedArguments parse_arguments(int argc, char **argv) {
  if (argc < 2)
    throw std::runtime_error(
        "usage: scanlan-mesh-repair <analyze|repair|version> [options]");
  ParsedArguments arguments;
  arguments.command = argv[1];
  if (arguments.command == "version") {
    if (argc != 3 || std::string(argv[2]) != "--json") {
      throw std::runtime_error("usage: scanlan-mesh-repair version --json");
    }
    arguments.json_version = true;
    return arguments;
  }
  if (arguments.command != "analyze" && arguments.command != "repair") {
    throw std::runtime_error("unknown command '" + arguments.command + "'");
  }
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    if (index + 1 >= argc)
      throw std::runtime_error("missing value for " + option);
    const std::string value = argv[++index];
    if (option == "--input") {
      arguments.input = value;
    } else if (option == "--policy") {
      arguments.policy = value;
    } else if (option == "--output") {
      arguments.output = value;
    } else if (option == "--report") {
      arguments.report = value;
    } else if (option == "--voxel-size-m") {
      std::size_t consumed = 0;
      arguments.voxel_size_m = std::stod(value, &consumed);
      if (consumed != value.size())
        throw std::runtime_error("invalid --voxel-size-m value");
    } else {
      throw std::runtime_error("unknown " + arguments.command + " option '" +
                               option + "'");
    }
  }
  if (arguments.input.empty())
    throw std::runtime_error(arguments.command + " requires --input");
  if (arguments.report.empty())
    throw std::runtime_error(arguments.command + " requires --report");
  if (arguments.command == "analyze" &&
      (!std::isfinite(arguments.voxel_size_m) ||
       arguments.voxel_size_m <= 0.0)) {
    throw std::runtime_error("--voxel-size-m must be a finite positive number");
  }
  if (arguments.command == "repair") {
    if (arguments.policy.empty())
      throw std::runtime_error("repair requires --policy");
    if (arguments.output.empty())
      throw std::runtime_error("repair requires --output");
  }
  if (fs::absolute(arguments.input).lexically_normal() ==
      fs::absolute(arguments.report).lexically_normal()) {
    throw std::runtime_error("--report must not overwrite --input");
  }
  if (!arguments.output.empty() &&
      fs::absolute(arguments.input).lexically_normal() ==
          fs::absolute(arguments.output).lexically_normal()) {
    throw std::runtime_error("--output must not overwrite --input");
  }
  return arguments;
}

void write_json(const fs::path &path, const Json &document) {
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path());
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot create report: " + path.string());
    output << document.dump(2) << '\n';
    if (!output)
      throw std::runtime_error("cannot write report: " + path.string());
  }
  std::error_code error;
  fs::remove(path, error);
  error.clear();
  fs::rename(temporary, path, error);
  if (error) {
    fs::remove(temporary);
    throw std::runtime_error("cannot finalize report: " + error.message());
  }
}

Json error_document(const std::string &command, const std::string &code,
                    const std::string &message) {
  return {{"schemaVersion", kSchemaVersion},
          {"algorithmVersion", kAlgorithmVersion},
          {"command", command},
          {"status", "error"},
          {"error", {{"code", code}, {"message", message}}}};
}

Json analyze(const ParsedArguments &arguments) {
  if (!fs::is_regular_file(arguments.input)) {
    throw std::runtime_error("input mesh does not exist or is not a file: " +
                             arguments.input.string());
  }
  const std::string fingerprint = sha256_file(arguments.input);
  std::vector<Point> raw_points;
  std::vector<std::vector<std::size_t>> polygons;
  std::string comments;
  if (!CGAL::IO::read_PLY(arguments.input.string(), raw_points, polygons,
                          comments)) {
    throw std::runtime_error("input is not a readable PLY polygon mesh");
  }
  if (raw_points.empty())
    throw std::runtime_error("input mesh contains no vertices");
  if (polygons.empty())
    throw std::runtime_error("input mesh contains no faces");

  std::map<PointKey, std::size_t> unique_point_indices;
  std::vector<Point> points;
  std::vector<std::size_t> canonical_vertex(raw_points.size());
  std::size_t duplicate_vertex_count = 0;
  for (std::size_t index = 0; index < raw_points.size(); ++index) {
    const auto values = point_array(raw_points[index]);
    if (!std::isfinite(values[0]) || !std::isfinite(values[1]) ||
        !std::isfinite(values[2])) {
      throw std::runtime_error(
          "input mesh contains a non-finite vertex coordinate");
    }
    const PointKey key = point_key(raw_points[index]);
    const auto [found, inserted] =
        unique_point_indices.emplace(key, points.size());
    if (inserted) {
      points.push_back(raw_points[index]);
    } else {
      ++duplicate_vertex_count;
    }
    canonical_vertex[index] = found->second;
  }

  std::vector<Face> faces;
  std::set<Face> seen_faces;
  std::size_t triangle_count = 0;
  std::size_t duplicate_face_count = 0;
  std::size_t degenerate_face_count = 0;
  for (std::size_t polygon_index = 0; polygon_index < polygons.size();
       ++polygon_index) {
    const auto &polygon = polygons[polygon_index];
    if (polygon.size() != 3) {
      throw std::runtime_error("input face " + std::to_string(polygon_index) +
                               " is not triangular");
    }
    Face face{};
    for (std::size_t corner = 0; corner < 3; ++corner) {
      if (polygon[corner] >= raw_points.size()) {
        throw std::runtime_error(
            "input face references an out-of-range vertex");
      }
      face[corner] = canonical_vertex[polygon[corner]];
    }
    ++triangle_count;
    const Face sorted = face_key(face);
    const bool duplicate = !seen_faces.insert(sorted).second;
    if (duplicate)
      ++duplicate_face_count;
    const bool degenerate =
        face[0] == face[1] || face[1] == face[2] || face[0] == face[2] ||
        CGAL::collinear(points[face[0]], points[face[1]], points[face[2]]);
    if (degenerate)
      ++degenerate_face_count;
    if (!duplicate && !degenerate)
      faces.push_back(face);
  }
  if (faces.empty())
    throw std::runtime_error("input mesh contains no analyzable triangles");

  std::map<Edge, std::vector<std::size_t>> edge_faces;
  std::vector<std::vector<std::size_t>> vertex_faces(points.size());
  for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
    const Face &face = faces[face_index];
    for (std::size_t corner = 0; corner < 3; ++corner) {
      edge_faces[edge_key(face[corner], face[(corner + 1) % 3])].push_back(
          face_index);
      vertex_faces[face[corner]].push_back(face_index);
    }
  }
  const std::size_t non_manifold_edge_count = static_cast<std::size_t>(
      std::count_if(edge_faces.begin(), edge_faces.end(),
                    [](const auto &entry) { return entry.second.size() > 2; }));

  std::size_t non_manifold_vertex_count = 0;
  for (std::size_t vertex = 0; vertex < vertex_faces.size(); ++vertex) {
    const auto &incident_faces = vertex_faces[vertex];
    if (incident_faces.empty())
      continue;
    std::map<std::size_t, std::size_t> local_index;
    for (std::size_t i = 0; i < incident_faces.size(); ++i)
      local_index[incident_faces[i]] = i;
    DisjointSet fan(incident_faces.size());
    std::size_t boundary_spokes = 0;
    bool bad_edge = false;
    std::set<Edge> spokes;
    for (const auto face_index : incident_faces) {
      const Face &face = faces[face_index];
      for (const auto other : face) {
        if (other != vertex)
          spokes.insert(edge_key(vertex, other));
      }
    }
    for (const Edge &spoke : spokes) {
      const auto &adjacent = edge_faces.at(spoke);
      if (adjacent.size() == 1) {
        ++boundary_spokes;
      } else if (adjacent.size() > 2) {
        bad_edge = true;
      }
      for (std::size_t i = 1; i < adjacent.size(); ++i) {
        fan.unite(local_index.at(adjacent[0]), local_index.at(adjacent[i]));
      }
    }
    std::set<std::size_t> fan_components;
    for (std::size_t i = 0; i < incident_faces.size(); ++i)
      fan_components.insert(fan.find(i));
    if (bad_edge || fan_components.size() != 1 ||
        (boundary_spokes != 0 && boundary_spokes != 2)) {
      ++non_manifold_vertex_count;
    }
  }

  DisjointSet face_components(faces.size());
  for (const auto &[edge, adjacent] : edge_faces) {
    (void)edge;
    for (std::size_t index = 1; index < adjacent.size(); ++index) {
      face_components.unite(adjacent.front(), adjacent[index]);
    }
  }
  std::map<std::size_t, std::vector<std::size_t>> grouped_faces;
  for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
    grouped_faces[face_components.find(face_index)].push_back(face_index);
  }
  std::vector<Component> components;
  std::vector<std::size_t> face_component(faces.size());
  for (auto &[root, component_faces] : grouped_faces) {
    (void)root;
    Component component;
    component.faces = component_faces;
    std::vector<std::string> signatures;
    for (const auto face_index : component.faces) {
      const Face &face = faces[face_index];
      face_component[face_index] = components.size();
      for (const auto vertex : face)
        component.vertices.insert(vertex);
      component.area +=
          triangle_area(points[face[0]], points[face[1]], points[face[2]]);
      std::array<std::string, 3> vertices = {
          stable_point_text(points[face[0]]),
          stable_point_text(points[face[1]]),
          stable_point_text(points[face[2]])};
      std::sort(vertices.begin(), vertices.end());
      signatures.push_back(vertices[0] + "|" + vertices[1] + "|" + vertices[2]);
    }
    std::sort(signatures.begin(), signatures.end());
    std::string identity;
    for (const auto &signature : signatures)
      identity += signature + ";";
    component.id = "component-" + sha256_text(identity).substr(0, 20);
    components.push_back(std::move(component));
  }

  std::size_t non_cycle_boundary_component_count = 0;
  auto loops = find_boundary_loops(edge_faces, points,
                                   non_cycle_boundary_component_count);
  std::vector<std::size_t> boundary_loop_counts(components.size(), 0);
  for (const auto &loop : loops)
    ++boundary_loop_counts[face_component[loop.face]];

  Json component_json = Json::array();
  std::vector<std::size_t> component_order(components.size());
  std::iota(component_order.begin(), component_order.end(), 0);
  std::sort(component_order.begin(), component_order.end(),
            [&](std::size_t a, std::size_t b) {
              return components[a].id < components[b].id;
            });
  for (const auto component_index : component_order) {
    const Component &component = components[component_index];
    component_json.push_back(
        {{"componentId", component.id},
         {"vertexCount", component.vertices.size()},
         {"triangleCount", component.faces.size()},
         {"surfaceAreaM2", component.area},
         {"boundaryLoopCount", boundary_loop_counts[component_index]},
         {"boundingBox", bounding_box_json(points, &component.vertices)}});
  }

  Mesh validation_mesh;
  std::vector<Mesh::Vertex_index> mesh_vertices;
  mesh_vertices.reserve(points.size());
  for (const Point &point : points)
    mesh_vertices.push_back(validation_mesh.add_vertex(point));
  std::size_t rejected_validation_faces = 0;
  for (const Face &face : faces) {
    if (validation_mesh.add_face(mesh_vertices[face[0]], mesh_vertices[face[1]],
                                 mesh_vertices[face[2]]) == Mesh::null_face()) {
      ++rejected_validation_faces;
    }
  }
  std::vector<std::pair<Mesh::Face_index, Mesh::Face_index>> intersections;
  if (validation_mesh.number_of_faces() > 1) {
    PMP::self_intersections(validation_mesh, std::back_inserter(intersections));
  }

  const Json mesh_bounds = bounding_box_json(points);
  Json loop_documents = Json::array();
  for (const auto &loop : loops) {
    const std::size_t component_index = face_component[loop.face];
    loop_documents.push_back(loop_json(loop, points, faces, edge_faces,
                                       components[component_index].id,
                                       mesh_bounds));
  }
  std::sort(loop_documents.begin(), loop_documents.end(),
            [](const Json &a, const Json &b) {
              return a.at("loopId").get<std::string>() <
                     b.at("loopId").get<std::string>();
            });

  Json warnings = Json::array();
  if (rejected_validation_faces > 0) {
    warnings.push_back("CGAL validation mesh rejected " +
                       std::to_string(rejected_validation_faces) +
                       " topologically incompatible triangle(s); raw topology "
                       "counts remain complete");
  }
  if (non_cycle_boundary_component_count > 0) {
    warnings.push_back(std::to_string(non_cycle_boundary_component_count) +
                       " boundary component(s) branch or terminate and are not "
                       "reported as loops");
  }

  return {{"schemaVersion", kSchemaVersion},
          {"algorithmVersion", kAlgorithmVersion},
          {"command", "analyze"},
          {"status", "ok"},
          {"inputMeshFingerprint",
           {{"algorithm", "sha256"}, {"value", fingerprint}}},
          {"voxelSizeM", arguments.voxel_size_m},
          {"mesh",
           {{"vertexCount", raw_points.size()},
            {"triangleCount", triangle_count},
            {"boundingBox", mesh_bounds}}},
          {"topology",
           {{"duplicateVertexCount", duplicate_vertex_count},
            {"duplicateTriangleCount", duplicate_face_count},
            {"degenerateTriangleCount", degenerate_face_count},
            {"nonManifoldEdgeCount", non_manifold_edge_count},
            {"nonManifoldVertexCount", non_manifold_vertex_count},
            {"selfIntersectionCount", intersections.size()},
            {"connectedComponentCount", components.size()},
            {"boundaryLoopCount", loops.size()},
            {"nonCycleBoundaryComponentCount",
             non_cycle_boundary_component_count}}},
          {"connectedComponents", std::move(component_json)},
          {"boundaryLoops", std::move(loop_documents)},
          {"warnings", std::move(warnings)}};
}

Json read_json_document(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open JSON file: " + path.string());
  try {
    return Json::parse(input);
  } catch (const Json::exception &error) {
    throw std::runtime_error("invalid JSON in " + path.string() + ": " +
                             error.what());
  }
}

struct RepairMesh {
  Mesh mesh;
  std::size_t raw_vertex_count = 0;
  std::size_t raw_triangle_count = 0;
  std::size_t duplicate_vertex_count = 0;
  std::size_t duplicate_triangle_count = 0;
  std::size_t degenerate_triangle_count = 0;
  std::size_t rejected_triangle_count = 0;
  std::size_t reoriented_triangle_count = 0;
  std::vector<PointKey> original_point_keys;
};

RepairMesh read_repair_mesh(const fs::path &path) {
  std::vector<Point> raw_points;
  std::vector<std::vector<std::size_t>> polygons;
  std::string comments;
  if (!CGAL::IO::read_PLY(path.string(), raw_points, polygons, comments))
    throw std::runtime_error("input is not a readable PLY polygon mesh");
  if (raw_points.empty() || polygons.empty())
    throw std::runtime_error("input mesh contains no geometry");

  RepairMesh result;
  result.raw_vertex_count = raw_points.size();
  result.raw_triangle_count = polygons.size();
  std::map<PointKey, std::size_t> unique_point_indices;
  std::vector<Point> points;
  std::vector<std::size_t> canonical_vertex(raw_points.size());
  for (std::size_t index = 0; index < raw_points.size(); ++index) {
    const auto values = point_array(raw_points[index]);
    if (!std::isfinite(values[0]) || !std::isfinite(values[1]) ||
        !std::isfinite(values[2]))
      throw std::runtime_error(
          "input mesh contains a non-finite vertex coordinate");
    const auto [found, inserted] = unique_point_indices.emplace(
        point_key(raw_points[index]), points.size());
    if (inserted)
      points.push_back(raw_points[index]);
    else
      ++result.duplicate_vertex_count;
    canonical_vertex[index] = found->second;
  }

  std::vector<Face> faces;
  std::set<Face> seen_faces;
  for (std::size_t polygon_index = 0; polygon_index < polygons.size();
       ++polygon_index) {
    const auto &polygon = polygons[polygon_index];
    if (polygon.size() != 3)
      throw std::runtime_error("input face " + std::to_string(polygon_index) +
                               " is not triangular");
    Face face{};
    for (std::size_t corner = 0; corner < 3; ++corner) {
      if (polygon[corner] >= raw_points.size())
        throw std::runtime_error(
            "input face references an out-of-range vertex");
      face[corner] = canonical_vertex[polygon[corner]];
    }
    if (!seen_faces.insert(face_key(face)).second) {
      ++result.duplicate_triangle_count;
      continue;
    }
    if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2] ||
        CGAL::collinear(points[face[0]], points[face[1]], points[face[2]])) {
      ++result.degenerate_triangle_count;
      continue;
    }
    faces.push_back(face);
  }
  if (faces.empty())
    throw std::runtime_error("input mesh contains no repairable triangles");

  std::vector<Mesh::Vertex_index> mesh_vertices;
  mesh_vertices.reserve(points.size());
  for (const Point &point : points)
    mesh_vertices.push_back(result.mesh.add_vertex(point));
  for (const Face &face : faces) {
    auto inserted = result.mesh.add_face(
        mesh_vertices[face[0]], mesh_vertices[face[1]], mesh_vertices[face[2]]);
    if (inserted == Mesh::null_face()) {
      inserted =
          result.mesh.add_face(mesh_vertices[face[0]], mesh_vertices[face[2]],
                               mesh_vertices[face[1]]);
      if (inserted == Mesh::null_face())
        ++result.rejected_triangle_count;
      else
        ++result.reoriented_triangle_count;
    }
  }
  PMP::remove_isolated_vertices(result.mesh);
  for (const auto vertex : result.mesh.vertices())
    result.original_point_keys.push_back(point_key(result.mesh.point(vertex)));
  if (result.mesh.number_of_faces() == 0)
    throw std::runtime_error("CGAL could not construct a repairable surface");
  return result;
}

std::size_t non_manifold_vertex_count(const Mesh &mesh) {
  std::size_t count = 0;
  for (const auto vertex : mesh.vertices()) {
    if (PMP::is_non_manifold_vertex(vertex, mesh))
      ++count;
  }
  return count;
}

std::size_t self_intersection_count(const Mesh &mesh) {
  if (mesh.number_of_faces() < 2)
    return 0;
  std::vector<std::pair<Mesh::Face_index, Mesh::Face_index>> intersections;
  PMP::self_intersections(mesh, std::back_inserter(intersections));
  return intersections.size();
}

std::vector<Point> border_points(const Mesh &mesh,
                                 Mesh::Halfedge_index border) {
  std::vector<Point> points;
  for (const auto halfedge : CGAL::halfedges_around_face(border, mesh))
    points.push_back(mesh.point(target(halfedge, mesh)));
  return points;
}

std::map<std::string, Mesh::Halfedge_index>
mesh_boundary_loops(const Mesh &mesh) {
  std::vector<Mesh::Halfedge_index> cycles;
  CGAL::extract_boundary_cycles(mesh, std::back_inserter(cycles));
  std::map<std::string, Mesh::Halfedge_index> result;
  for (const auto cycle : cycles) {
    const std::string id = loop_id_from_points(border_points(mesh, cycle));
    if (!result.emplace(id, cycle).second)
      throw std::runtime_error(
          "mesh contains duplicate geometric boundary IDs");
  }
  return result;
}

Vector mesh_face_normal(const Mesh &mesh, Mesh::Face_index face_index) {
  const auto first = halfedge(face_index, mesh);
  const Point &a = mesh.point(source(first, mesh));
  const Point &b = mesh.point(target(first, mesh));
  const Point &c = mesh.point(target(next(first, mesh), mesh));
  const Vector normal = CGAL::cross_product(b - a, c - a);
  const double length = vector_length(normal);
  return length > 0.0 ? normal / length : Vector(0.0, 0.0, 0.0);
}

double mesh_face_area(const Mesh &mesh, Mesh::Face_index face_index) {
  const auto first = halfedge(face_index, mesh);
  return triangle_area(mesh.point(source(first, mesh)),
                       mesh.point(target(first, mesh)),
                       mesh.point(target(next(first, mesh), mesh)));
}

double
seam_discontinuity_degrees(const Mesh &mesh,
                           const std::set<Mesh::Face_index> &patch_faces) {
  double maximum = 0.0;
  for (const auto patch_face : patch_faces) {
    const Vector patch_normal = mesh_face_normal(mesh, patch_face);
    for (const auto halfedge :
         CGAL::halfedges_around_face(halfedge(patch_face, mesh), mesh)) {
      const auto neighbor = face(opposite(halfedge, mesh), mesh);
      if (neighbor == Mesh::null_face() || patch_faces.contains(neighbor))
        continue;
      const Vector neighbor_normal = mesh_face_normal(mesh, neighbor);
      const double dot = std::clamp(
          CGAL::to_double(patch_normal * neighbor_normal), -1.0, 1.0);
      maximum = std::max(maximum, std::acos(dot) * 180.0 / CGAL_PI);
    }
  }
  return maximum;
}

void write_mesh(const fs::path &path, const Mesh &mesh) {
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path());
  const fs::path temporary = path.string() + ".tmp.ply";
  if (!CGAL::IO::write_PLY(temporary.string(), mesh,
                           CGAL::parameters::use_binary_mode(false))) {
    throw std::runtime_error("cannot write repaired PLY mesh");
  }
  std::error_code error;
  fs::remove(path, error);
  error.clear();
  fs::rename(temporary, path, error);
  if (error) {
    fs::remove(temporary);
    throw std::runtime_error("cannot finalize repaired mesh: " +
                             error.message());
  }
}

Json repair(const ParsedArguments &arguments) {
  if (!fs::is_regular_file(arguments.input))
    throw std::runtime_error("input mesh does not exist or is not a file: " +
                             arguments.input.string());
  if (!fs::is_regular_file(arguments.policy))
    throw std::runtime_error("repair policy does not exist or is not a file: " +
                             arguments.policy.string());
  const std::string fingerprint = sha256_file(arguments.input);
  const Json policy = read_json_document(arguments.policy);
  if (policy.value("schemaVersion", 0) != kSchemaVersion)
    throw std::runtime_error("repair policy schemaVersion is unsupported");
  if (policy.value("algorithmVersion", "") != kAlgorithmVersion)
    throw std::runtime_error(
        "repair policy algorithmVersion does not match the backend");
  if (!policy.contains("inputMeshFingerprint"))
    throw std::runtime_error("repair policy has no input fingerprint");
  const Json &policy_fingerprint = policy.at("inputMeshFingerprint");
  const std::string expected_fingerprint =
      policy_fingerprint.is_object() ? policy_fingerprint.value("value", "")
                                     : policy_fingerprint.get<std::string>();
  if (expected_fingerprint != fingerprint)
    throw std::runtime_error(
        "repair policy is stale or belongs to a different input mesh");
  const std::string profile = policy.value("profile", "faithful");
  if (profile != "faithful" && profile != "architectural" &&
      profile != "natural")
    throw std::runtime_error("repair policy profile is unsupported: " +
                             profile);

  RepairMesh input = read_repair_mesh(arguments.input);
  Mesh &mesh = input.mesh;
  const std::size_t before_non_manifold = non_manifold_vertex_count(mesh);
  const std::size_t before_intersections = self_intersection_count(mesh);
  const std::size_t stitched_border_pairs = PMP::stitch_borders(mesh);
  std::size_t duplicated_non_manifold_vertices = 0;
  if (policy.value("repairNonManifold", true))
    duplicated_non_manifold_vertices =
        PMP::duplicate_non_manifold_vertices(mesh);
  if (policy.value("repairSelfIntersections", false) &&
      self_intersection_count(mesh) > 0) {
    PMP::autorefine(mesh,
                    CGAL::parameters::apply_iterative_snap_rounding(true));
  }

  auto available_loops = mesh_boundary_loops(mesh);
  struct SelectedLoop {
    std::string id;
    Json document;
  };
  std::vector<SelectedLoop> selected;
  for (const Json &entry : policy.value("selectedLoops", Json::array())) {
    if (!entry.is_object())
      throw std::runtime_error("selectedLoops entries must be objects");
    const std::string classification = entry.value("classification", "");
    if (classification != "fill_measured" && classification != "fill_inferred")
      throw std::runtime_error(
          "policy attempted to authorize a non-fill classification");
    selected.push_back({entry.at("loopId").get<std::string>(), entry});
  }
  std::sort(
      selected.begin(), selected.end(),
      [](const SelectedLoop &a, const SelectedLoop &b) { return a.id < b.id; });
  if (std::adjacent_find(selected.begin(), selected.end(),
                         [](const SelectedLoop &a, const SelectedLoop &b) {
                           return a.id == b.id;
                         }) != selected.end())
    throw std::runtime_error("repair policy contains a duplicate loop ID");

  Json filled_loops = Json::array();
  for (const SelectedLoop &selection : selected) {
    const auto found = available_loops.find(selection.id);
    if (found == available_loops.end())
      throw std::runtime_error(
          "authorized loop is absent from the input mesh: " + selection.id);
    std::vector<Mesh::Face_index> patch_faces;
    std::vector<Mesh::Vertex_index> patch_vertices;
    if (profile == "faithful") {
      PMP::triangulate_hole(mesh, found->second,
                            CGAL::parameters::face_output_iterator(
                                std::back_inserter(patch_faces)));
    } else if (profile == "architectural") {
      PMP::triangulate_and_refine_hole(mesh, found->second,
                                       std::back_inserter(patch_faces),
                                       std::back_inserter(patch_vertices));
      const Json &plane_json = selection.document.at("bestFitPlane");
      const auto origin = plane_json.at("origin").get<std::array<double, 3>>();
      const auto normal = plane_json.at("normal").get<std::array<double, 3>>();
      const Plane plane(Point(origin[0], origin[1], origin[2]),
                        Vector(normal[0], normal[1], normal[2]));
      for (const auto vertex : patch_vertices)
        mesh.point(vertex) = plane.projection(mesh.point(vertex));
    } else {
      const auto result = PMP::triangulate_refine_and_fair_hole(
          mesh, found->second, std::back_inserter(patch_faces),
          std::back_inserter(patch_vertices));
      if (!std::get<0>(result))
        throw std::runtime_error("natural patch fairing failed for " +
                                 selection.id);
    }
    if (patch_faces.empty())
      throw std::runtime_error("CGAL could not triangulate authorized loop " +
                               selection.id);
    const std::set<Mesh::Face_index> patch_set(patch_faces.begin(),
                                               patch_faces.end());
    double area = 0.0;
    for (const auto face_index : patch_faces)
      area += mesh_face_area(mesh, face_index);
    filled_loops.push_back(
        {{"loopId", selection.id},
         {"classification", selection.document.at("classification")},
         {"areaAddedM2", area},
         {"triangleCountAdded", patch_faces.size()},
         {"vertexCountAdded", patch_vertices.size()},
         {"maximumSeamNormalDiscontinuityDegrees",
          seam_discontinuity_degrees(mesh, patch_set)}});
  }

  PMP::remove_isolated_vertices(mesh);
  std::set<PointKey> output_points;
  for (const auto vertex : mesh.vertices())
    output_points.insert(point_key(mesh.point(vertex)));
  for (const auto &original : input.original_point_keys) {
    if (!output_points.contains(original))
      throw std::runtime_error(
          "repair moved or removed an original valid vertex");
  }
  const std::size_t after_non_manifold = non_manifold_vertex_count(mesh);
  if (after_non_manifold > before_non_manifold)
    throw std::runtime_error("repair increased non-manifold vertex count");
  const std::size_t after_intersections = self_intersection_count(mesh);
  const auto remaining_loops = mesh_boundary_loops(mesh);
  write_mesh(arguments.output, mesh);

  return {
      {"schemaVersion", kSchemaVersion},
      {"algorithmVersion", kAlgorithmVersion},
      {"command", "repair"},
      {"status", "ok"},
      {"profile", profile},
      {"inputMeshFingerprint",
       {{"algorithm", "sha256"}, {"value", fingerprint}}},
      {"outputMeshFingerprint",
       {{"algorithm", "sha256"}, {"value", sha256_file(arguments.output)}}},
      {"operations",
       {{"duplicateVerticesRemoved", input.duplicate_vertex_count},
        {"duplicateTrianglesRemoved", input.duplicate_triangle_count},
        {"degenerateTrianglesRemoved", input.degenerate_triangle_count},
        {"topologicallyIncompatibleTrianglesRemoved",
         input.rejected_triangle_count},
        {"trianglesReoriented", input.reoriented_triangle_count},
        {"coincidentBorderPairsStitched", stitched_border_pairs},
        {"nonManifoldVerticesDuplicated", duplicated_non_manifold_vertices}}},
      {"topologyBefore",
       {{"vertexCount", input.raw_vertex_count},
        {"triangleCount", input.raw_triangle_count},
        {"nonManifoldVertexCount", before_non_manifold},
        {"selfIntersectionCount", before_intersections},
        {"boundaryLoopCount", available_loops.size()}}},
      {"topologyAfter",
       {{"vertexCount", mesh.number_of_vertices()},
        {"triangleCount", mesh.number_of_faces()},
        {"nonManifoldVertexCount", after_non_manifold},
        {"selfIntersectionCount", after_intersections},
        {"boundaryLoopCount", remaining_loops.size()}}},
      {"filledLoops", std::move(filled_loops)},
      {"unauthorizedLoopFillCount", 0},
      {"originalVertexMaximumDisplacementM", 0.0}};
}

} // namespace

int main(int argc, char **argv) {
  std::optional<ParsedArguments> arguments;
  try {
    arguments = parse_arguments(argc, argv);
    if (arguments->command == "version") {
      const Json version = {
          {"schemaVersion", kSchemaVersion},
          {"algorithmVersion", kAlgorithmVersion},
          {"backend", {{"name", "CGAL"}, {"version", CGAL_VERSION_STR}}}};
      std::cout << version.dump() << '\n';
      return 0;
    }
    write_json(arguments->report, arguments->command == "analyze"
                                      ? analyze(*arguments)
                                      : repair(*arguments));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "scanlan-mesh-repair: " << error.what() << '\n';
    if (arguments &&
        (arguments->command == "analyze" || arguments->command == "repair") &&
        !arguments->report.empty()) {
      try {
        write_json(arguments->report,
                   error_document(arguments->command,
                                  arguments->command == "analyze"
                                      ? "analysis_failed"
                                      : "repair_failed",
                                  error.what()));
      } catch (const std::exception &report_error) {
        std::cerr << "scanlan-mesh-repair: failed to write JSON error report: "
                  << report_error.what() << '\n';
      }
    }
    return 2;
  }
}
