import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.183.2/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.183.2/examples/jsm/controls/OrbitControls.js";

const STATUS_COLORS = {
  Query: 0xffffff,
  Returned: 0xf7c948,
  Matched: 0x57d68d,
  FilteredOut: 0x8b95a7,
  Tombstoned: 0xff6b6b,
  Visited: 0x64d2ff,
};

const NODE_RADIUS = {
  Query: 0.18,
  Returned: 0.15,
  Matched: 0.12,
  FilteredOut: 0.09,
  Tombstoned: 0.11,
  Visited: 0.1,
};

const canvas = document.querySelector("#trace-canvas");
const summary = document.querySelector("#trace-summary");
const planList = document.querySelector("#plan-list");
const legend = document.querySelector("#legend");
const nodeDetails = document.querySelector("#node-details");
const fileInput = document.querySelector("#trace-file");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x111214);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x111214, 8, 28);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
camera.position.set(0, 3.6, 8.2);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const graphRoot = new THREE.Group();
scene.add(graphRoot);

const ambient = new THREE.AmbientLight(0xffffff, 0.75);
scene.add(ambient);
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(3, 6, 5);
scene.add(keyLight);

let currentTrace = null;
let currentPlan = null;
let nodeMeshes = [];

init();

async function init() {
  renderLegend();
  fileInput.addEventListener("change", handleFile);
  canvas.addEventListener("pointerdown", handlePick);
  window.addEventListener("resize", resize);
  resize();
  animate();

  const response = await fetch("./sample-trace.json");
  const trace = await response.json();
  loadTrace(trace);
}

function loadTrace(trace) {
  currentTrace = normalizeTrace(trace);
  currentPlan = currentTrace.per_plan_paths[0]?.plan ?? null;
  renderPlanButtons();
  renderGraph();
  renderSummary();
}

function normalizeTrace(trace) {
  return {
    version: trace.version ?? 1,
    title: trace.title ?? "Trace graph",
    query: trace.query ?? {},
    nodes: trace.nodes ?? [],
    edges: trace.edges ?? [],
    per_plan_paths: trace.per_plan_paths ?? [],
    truncated: Boolean(trace.truncated),
  };
}

function renderGraph() {
  graphRoot.clear();
  nodeMeshes = [];

  const nodes = currentTrace.nodes.map((node, index) => ({
    ...node,
    position: initialPosition(node.id, index, currentTrace.nodes.length),
  }));
  relax(nodes, currentTrace.edges);
  const nodeById = new Map(nodes.map((node) => [node.id, node]));

  const edgeGeometry = new THREE.BufferGeometry();
  const edgePositions = [];
  for (const edge of currentTrace.edges) {
    const from = nodeById.get(edge.from);
    const to = nodeById.get(edge.to);
    if (!from || !to) {
      continue;
    }
    edgePositions.push(...from.position.toArray(), ...to.position.toArray());
  }
  edgeGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(edgePositions, 3),
  );
  const edges = new THREE.LineSegments(
    edgeGeometry,
    new THREE.LineBasicMaterial({
      color: 0x3a4250,
      transparent: true,
      opacity: 0.42,
    }),
  );
  graphRoot.add(edges);

  for (const node of nodes) {
    const mesh = makeNode(node);
    graphRoot.add(mesh);
    nodeMeshes.push(mesh);
  }

  renderPath(nodeById);
}

function makeNode(node) {
  const status = node.status ?? "Visited";
  const geometry = new THREE.SphereGeometry(NODE_RADIUS[status] ?? 0.1, 24, 16);
  const material = new THREE.MeshStandardMaterial({
    color: STATUS_COLORS[status] ?? STATUS_COLORS.Visited,
    roughness: 0.45,
    metalness: 0.05,
    emissive: STATUS_COLORS[status] ?? STATUS_COLORS.Visited,
    emissiveIntensity: status === "Query" || status === "Returned" ? 0.22 : 0.05,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(node.position);
  mesh.userData.node = node;
  return mesh;
}

function renderPath(nodeById) {
  const path = currentTrace.per_plan_paths.find((plan) => plan.plan === currentPlan);
  if (!path || path.ordered_events.length < 2) {
    return;
  }

  const pathPositions = [];
  for (const id of path.ordered_events) {
    const node = nodeById.get(id);
    if (node) {
      pathPositions.push(...node.position.toArray());
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(pathPositions, 3),
  );
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({
      color: 0x4fb3ff,
      linewidth: 2,
      transparent: true,
      opacity: 0.96,
    }),
  );
  graphRoot.add(line);
}

function renderPlanButtons() {
  planList.replaceChildren();
  for (const plan of currentTrace.per_plan_paths) {
    const button = document.createElement("button");
    button.className = "plan-button";
    button.type = "button";
    button.setAttribute("aria-pressed", String(plan.plan === currentPlan));
    const label = document.createElement("span");
    label.textContent = plan.plan;
    const steps = document.createElement("small");
    steps.textContent = `${plan.ordered_events.length} steps`;
    button.append(label, steps);
    button.addEventListener("click", () => {
      currentPlan = plan.plan;
      renderPlanButtons();
      renderGraph();
    });
    planList.append(button);
  }
}

function renderLegend() {
  legend.replaceChildren();
  for (const [status, color] of Object.entries(STATUS_COLORS)) {
    const row = document.createElement("div");
    row.className = "legend-row";
    row.innerHTML = `<span class="swatch" style="background:#${color.toString(16).padStart(6, "0")}"></span><span>${status}</span>`;
    legend.append(row);
  }
}

function renderSummary() {
  const query = currentTrace.query;
  const parts = [
    `${currentTrace.nodes.length} nodes`,
    `${currentTrace.edges.length} edges`,
    `${currentTrace.per_plan_paths.length} plan overlays`,
  ];
  if (query.filter) {
    parts.push(`filter ${query.filter}`);
  }
  if (query.correlation) {
    parts.push(query.correlation);
  }
  summary.textContent = `${currentTrace.title}. ${parts.join(", ")}.`;
}

function initialPosition(id, index, total) {
  const angle = index * 2.399963229728653;
  const radius = 1.3 + (index / Math.max(total - 1, 1)) * 2.7;
  const y = ((id % 11) - 5) * 0.12;
  return new THREE.Vector3(
    Math.cos(angle) * radius,
    y,
    Math.sin(angle) * radius,
  );
}

function relax(nodes, edges) {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  for (let tick = 0; tick < 95; tick += 1) {
    for (const edge of edges) {
      const from = nodeById.get(edge.from);
      const to = nodeById.get(edge.to);
      if (!from || !to) {
        continue;
      }
      const delta = to.position.clone().sub(from.position);
      const distance = Math.max(delta.length(), 0.001);
      const pull = (distance - 1.0) * 0.014;
      delta.normalize().multiplyScalar(pull);
      from.position.add(delta);
      to.position.sub(delta);
    }
  }
}

function handlePick(event) {
  const bounds = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
  pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(nodeMeshes, false);
  if (hits.length === 0) {
    return;
  }
  const node = hits[0].object.userData.node;
  nodeDetails.textContent = JSON.stringify(
    {
      id: node.id,
      status: node.status,
      label: node.label,
      plan_step: planStep(node.id),
    },
    null,
    2,
  );
}

function planStep(id) {
  const path = currentTrace.per_plan_paths.find((plan) => plan.plan === currentPlan);
  const index = path?.ordered_events.indexOf(id) ?? -1;
  return index >= 0 ? index + 1 : null;
}

async function handleFile(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  loadTrace(JSON.parse(text));
}

function resize() {
  const parent = canvas.parentElement;
  const width = parent.clientWidth;
  const height = parent.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
