/**
 * DRONERL — Three.js 3D Quadrotor Course Renderer
 * Renders: drone (body + arms + rotors), obstacle course, gates, LIDAR,
 *          projectiles, wind particles, thermal zones.
 *
 * Coordinate convention:
 *   Physics (drone_environment.py): X=forward, Y=right, Z=up
 *   Three.js: X=right, Y=up, Z=forward
 *   Mapping: threePos = (physics.x, physics.z, physics.y)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ── Constants ─────────────────────────────────────────────────
const COURSE_W = 40.0;
const COURSE_D = 40.0;
const COURSE_H = 20.0;
const ARM_LENGTH = 0.125;
const MAX_RPM = 12000;
const MAX_LIDAR_RANGE = 10.0;
const LERP_SPEED = 0.25;

// Colors — amber/orange military theme
const AMBER = new THREE.Color(0xff9900);
const GATE_GREEN = new THREE.Color(0x00ff88);
const GATE_GOLD = new THREE.Color(0xffd700);
const GATE_GRAY = new THREE.Color(0x556666);
const PROJ_RED = new THREE.Color(0xff2200);
const WIND_CYAN = new THREE.Color(0x44aaff);
const WALL_COLOR = new THREE.Color(0x445566);
const COLUMN_COLOR = new THREE.Color(0x556677);
const THERMAL_COLOR = new THREE.Color(0xff4400);

// LIDAR ray directions in body frame (12 rays) — must match environment
const LIDAR_DIRS = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [0.707, 0.707, 0], [-0.707, 0.707, 0],
    [0.707, -0.707, 0], [-0.707, -0.707, 0],
    [0, 0, 1], [0, 0, -1],
    [0.707, 0, 0.5], [0.707, 0, -0.5],
];

// ── Module state ──────────────────────────────────────────────
let scene, camera, renderer, composer, controls;
let droneGroup;
let propGroups = [];
let guardRings = [];
let armLEDs = [];
let lidarLines = [];
let gateFrames = [];
let obstacleMeshes = [];
let movingObstacles = [];    // { mesh, data } for animated walls
let thermalEmitters = [];
let ewDomes = [];
let projectileMeshes = [];
let windParticles = null;
let courseBounds;
let droneShadow;
let courseLayout = null;
let taskTargetMarker = null;  // Visual for fly_to_point / altitude targets
let adversaryGroup = null;    // Red adversary drone mesh
let adversaryTrail = null;    // Fading red trail
let turretMarker = null;      // Fixed turret position marker (user weapon)
let turretMeshes = [];        // Course gun installations [{group, barrel, flash}]
let _weaponMode = false;      // Whether weapon crosshair is active
let _crosshairEl = null;      // CSS crosshair overlay element

// Swarm visualization (all 64 training envs)
let swarmDrones = [];         // Array of THREE.Group meshes (N-1 ghosts, skip env[0])
let swarmCurrents = [];       // Lerp state per ghost: {x,y,z,qw,qx,qy,qz}
let swarmTargets = [];        // Raw data from server: [[x,y,z,qw,qx,qy,qz], ...]
let _swarmVisible = false;
let _swarmBuilt = false;

// Arm tip positions in local drone space (for ground clamp calculation)
// These are the lowest-reaching points when the drone tilts
const ARM_TIP_LOCALS = [];  // populated in _buildDrone()

// Target / current state for interpolation
let targetDrone = null;
let currentDrone = { x: 0, y: 0, z: 1, qw: 1, qx: 0, qy: 0, qz: 0 };
let latestMsg = null;
let animTime = 0;

// Chase camera state
let _chaseMode = false;
const CHASE_DIST = 4.0;    // distance behind drone
const CHASE_HEIGHT = 2.0;  // height above drone
const CHASE_LERP = 0.06;   // camera smoothing (lower = smoother)

// ── Helper: physics → Three.js coordinate swap ───────────────
function p2t(px, py, pz) {
    return new THREE.Vector3(px, pz, py);
}

// ── Initialization ────────────────────────────────────────────
function init(container) {
    const w = container.clientWidth;
    const h = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1018);
    scene.fog = new THREE.FogExp2(0x0e1018, 0.008);

    camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 200);
    camera.position.set(15, 15, 15);
    camera.lookAt(0, 5, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.3, 0.5, 0.9);
    composer.addPass(bloom);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(COURSE_W / 2, 5, COURSE_D / 2);
    controls.maxDistance = 80;

    // Lighting
    scene.add(new THREE.AmbientLight(0x445566, 1.0));
    const dirLight = new THREE.DirectionalLight(0xfff4e0, 1.2);
    dirLight.position.set(20, 30, 20);
    dirLight.castShadow = true;
    scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0x334466, 0.3);
    fillLight.position.set(-10, -5, 10);
    scene.add(fillLight);

    _buildGround();
    _buildCourseBounds();
    _buildDrone();
    _buildLidar();
    _buildWindParticles();

    window.addEventListener('resize', () => {
        const nw = container.clientWidth;
        const nh = container.clientHeight;
        camera.aspect = nw / nh;
        camera.updateProjectionMatrix();
        renderer.setSize(nw, nh);
        composer.setSize(nw, nh);
    });

    _animate();
}

// ── Ground Plane ──────────────────────────────────────────────
function _buildGround() {
    const gridHelper = new THREE.GridHelper(80, 80, 0x3d2800, 0x1a1200);
    scene.add(gridHelper);

    const groundGeo = new THREE.PlaneGeometry(100, 100);
    const groundMat = new THREE.MeshStandardMaterial({
        color: 0x141418, roughness: 0.9, metalness: 0.1,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    scene.add(ground);
}

// ── Course Bounds ─────────────────────────────────────────────
function _buildCourseBounds() {
    const geo = new THREE.BoxGeometry(COURSE_W, COURSE_H, COURSE_D);
    const edges = new THREE.EdgesGeometry(geo);
    const mat = new THREE.LineBasicMaterial({
        color: AMBER, transparent: true, opacity: 0.2,
    });
    courseBounds = new THREE.LineSegments(edges, mat);
    // Center the box so it goes from (0,0,0) to (W,H,D)
    courseBounds.position.set(COURSE_W / 2, COURSE_H / 2, COURSE_D / 2);
    scene.add(courseBounds);
}

// ── Drone Model ───────────────────────────────────────────────
function _buildDrone() {
    droneGroup = new THREE.Group();

    // Central body — flattened octahedron
    const bodyGeo = new THREE.OctahedronGeometry(0.12, 0);
    bodyGeo.scale(1, 0.5, 1);
    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0x333333, metalness: 0.8, roughness: 0.3,
        emissive: 0x221100, emissiveIntensity: 0.4,
    });
    droneGroup.add(new THREE.Mesh(bodyGeo, bodyMat));

    const armAngles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4];
    const armMat = new THREE.MeshStandardMaterial({ color: 0x444444, metalness: 0.6, roughness: 0.4 });
    const guardMat = new THREE.MeshStandardMaterial({
        color: 0x555555, metalness: 0.5, roughness: 0.5, transparent: true, opacity: 0.6,
    });
    const bladeMat = new THREE.MeshStandardMaterial({
        color: 0x888888, metalness: 0.7, roughness: 0.3, side: THREE.DoubleSide,
    });
    const motorMat = new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.9, roughness: 0.2 });

    propGroups = [];
    guardRings = [];
    armLEDs = [];
    const scaledArm = ARM_LENGTH * 8;
    const propRadius = 0.14;
    const guardRadius = 0.16;

    for (let i = 0; i < 4; i++) {
        const angle = armAngles[i];
        const dx = Math.cos(angle) * scaledArm;
        const dz = Math.sin(angle) * scaledArm;

        // Arm beam
        const armGeo = new THREE.CylinderGeometry(0.02, 0.02, scaledArm, 6);
        armGeo.rotateZ(Math.PI / 2);
        const arm = new THREE.Mesh(armGeo, armMat);
        arm.position.set(dx / 2, 0, dz / 2);
        arm.rotation.y = -angle;
        droneGroup.add(arm);

        // Motor housing
        const motorGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.04, 12);
        const motor = new THREE.Mesh(motorGeo, motorMat);
        motor.position.set(dx, 0.01, dz);
        droneGroup.add(motor);

        // Prop guard ring (stationary)
        const guardGeo = new THREE.TorusGeometry(guardRadius, 0.008, 6, 24);
        guardGeo.rotateX(Math.PI / 2);
        const guard = new THREE.Mesh(guardGeo, guardMat.clone());
        guard.position.set(dx, 0.03, dz);
        droneGroup.add(guard);
        guardRings.push(guard);

        // Propeller blades (spinning group)
        const propGroup = new THREE.Group();
        propGroup.position.set(dx, 0.035, dz);
        for (let b = 0; b < 2; b++) {
            const bladeGeo = new THREE.BoxGeometry(propRadius * 2, 0.004, 0.025);
            const blade = new THREE.Mesh(bladeGeo, bladeMat.clone());
            blade.rotation.y = b * Math.PI / 2;
            blade.rotation.z = 0.08;
            propGroup.add(blade);
        }
        droneGroup.add(propGroup);
        propGroups.push(propGroup);

        // LED
        const led = new THREE.PointLight(0xff9900, 0.8, 4);
        led.position.set(dx, -0.05, dz);
        droneGroup.add(led);
        armLEDs.push(led);

        // Track extremity points for ground clamp (arm tips + LED positions)
        ARM_TIP_LOCALS.push(new THREE.Vector3(dx, -0.05, dz));
    }
    // Body bottom
    ARM_TIP_LOCALS.push(new THREE.Vector3(0, -0.06, 0));

    // Shadow
    const shadowGeo = new THREE.CircleGeometry(0.4, 16);
    const shadowMat = new THREE.MeshBasicMaterial({
        color: 0x000000, transparent: true, opacity: 0.35,
    });
    droneShadow = new THREE.Mesh(shadowGeo, shadowMat);
    droneShadow.rotation.x = -Math.PI / 2;
    droneShadow.position.y = 0.02;
    scene.add(droneShadow);

    droneGroup.position.set(2, 2, COURSE_D / 2);
    scene.add(droneGroup);
}

// ── Swarm Ghost Drones ───────────────────────────────────────

function _buildSwarmDrone() {
    const group = new THREE.Group();

    // Body — smaller flattened octahedron, semi-transparent amber
    const bodyGeo = new THREE.OctahedronGeometry(0.08, 0);
    bodyGeo.scale(1, 0.5, 1);
    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0xff9900, metalness: 0.4, roughness: 0.5,
        emissive: 0x331100, emissiveIntensity: 0.6,
        transparent: true, opacity: 0.4,
    });
    group.add(new THREE.Mesh(bodyGeo, bodyMat));

    // 4 stub arms — half-length, thinner
    const armAngles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4];
    const armMat = new THREE.MeshStandardMaterial({
        color: 0x666666, transparent: true, opacity: 0.3,
    });
    const stubLen = ARM_LENGTH * 4;
    for (let i = 0; i < 4; i++) {
        const angle = armAngles[i];
        const dx = Math.cos(angle) * stubLen;
        const dz = Math.sin(angle) * stubLen;
        const armGeo = new THREE.CylinderGeometry(0.012, 0.012, stubLen, 4);
        armGeo.rotateZ(Math.PI / 2);
        const arm = new THREE.Mesh(armGeo, armMat);
        arm.position.set(dx / 2, 0, dz / 2);
        arm.rotation.y = -angle;
        group.add(arm);
    }

    return group;
}

function _initSwarm(nEnvs) {
    if (_swarmBuilt) return;
    _swarmBuilt = true;

    // Create ghost drones for envs 1..N-1 (skip env[0] — that's the primary)
    const nGhosts = (nEnvs || 64) - 1;
    for (let i = 0; i < nGhosts; i++) {
        const ghost = _buildSwarmDrone();
        ghost.visible = _swarmVisible;
        scene.add(ghost);
        swarmDrones.push(ghost);
        swarmCurrents.push({ x: 0, y: 0, z: 1, qw: 1, qx: 0, qy: 0, qz: 0 });
    }
}

function setSwarmMode(enabled) {
    _swarmVisible = enabled;
    for (const ghost of swarmDrones) {
        ghost.visible = enabled;
    }
}

// ── LIDAR Visualization ───────────────────────────────────────
function _buildLidar() {
    lidarLines = [];
    const mat = new THREE.LineBasicMaterial({
        color: 0x00ff44, transparent: true, opacity: 0.4,
    });
    for (let i = 0; i < 12; i++) {
        const geo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(1, 0, 0),
        ]);
        const line = new THREE.Line(geo, mat.clone());
        line.visible = false;
        scene.add(line);
        lidarLines.push(line);
    }
}

// ── Wind Particles ────────────────────────────────────────────
function _buildWindParticles() {
    const count = 300;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        positions[i * 3] = Math.random() * COURSE_W;
        positions[i * 3 + 1] = Math.random() * COURSE_H;
        positions[i * 3 + 2] = Math.random() * COURSE_D;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
        color: WIND_CYAN.getHex(), size: 0.08,
        transparent: true, opacity: 0.15,
        blending: THREE.AdditiveBlending,
    });
    windParticles = new THREE.Points(geo, mat);
    windParticles.visible = false;
    scene.add(windParticles);
}

// ══════════════════════════════════════════════════════════════
//  Course Building (called when course_layout message arrives)
// ══════════════════════════════════════════════════════════════

function buildCourse(layout) {
    courseLayout = layout;

    // Clear previous course elements
    obstacleMeshes.forEach(m => scene.remove(m));
    obstacleMeshes = [];
    movingObstacles = [];
    gateFrames.forEach(m => scene.remove(m));
    gateFrames = [];
    thermalEmitters.forEach(m => scene.remove(m));
    thermalEmitters = [];
    ewDomes.forEach(m => scene.remove(m));
    ewDomes = [];
    turretMeshes.forEach(t => scene.remove(t.group));
    turretMeshes = [];
    if (taskTargetMarker) { scene.remove(taskTargetMarker); taskTargetMarker = null; }

    // ── Obstacles ─────────────────────────────────────────────
    const wallMat = new THREE.MeshStandardMaterial({
        color: WALL_COLOR, metalness: 0.4, roughness: 0.6,
        transparent: true, opacity: 0.35,
    });
    const wallEdgeMat = new THREE.LineBasicMaterial({
        color: 0x88aacc, transparent: true, opacity: 0.6,
    });
    const colMat = new THREE.MeshStandardMaterial({
        color: COLUMN_COLOR, metalness: 0.3, roughness: 0.7,
        transparent: true, opacity: 0.4,
    });
    const movingMat = new THREE.MeshStandardMaterial({
        color: 0xff6633, metalness: 0.4, roughness: 0.6,
        transparent: true, opacity: 0.3, emissive: 0xff3300, emissiveIntensity: 0.15,
    });

    for (const obs of (layout.obstacles || [])) {
        if (obs.type === 'wall' || obs.type === 'moving') {
            const mn = obs.min;
            const mx = obs.max;
            const sx = mx[0] - mn[0];
            const sy = mx[1] - mn[1];
            const sz = mx[2] - mn[2];
            // Physics→Three.js: (x, z, y), size (sx, sz, sy)
            const geo = new THREE.BoxGeometry(sx, sz, sy);
            const mat = obs.type === 'moving' ? movingMat.clone() : wallMat.clone();
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.copy(p2t(
                mn[0] + sx / 2, mn[1] + sy / 2, mn[2] + sz / 2
            ));
            scene.add(mesh);
            obstacleMeshes.push(mesh);

            // Wireframe edges for visibility
            const edges = new THREE.LineSegments(
                new THREE.EdgesGeometry(geo),
                obs.type === 'moving'
                    ? new THREE.LineBasicMaterial({ color: 0xff6633, transparent: true, opacity: 0.6 })
                    : wallEdgeMat.clone()
            );
            edges.position.copy(mesh.position);
            scene.add(edges);
            obstacleMeshes.push(edges);

            if (obs.type === 'moving') {
                movingObstacles.push({ mesh, edges, data: obs });
            }
        } else if (obs.type === 'column') {
            const height = obs.z_max - obs.z_min;
            const geo = new THREE.CylinderGeometry(obs.radius, obs.radius, height, 16);
            const mesh = new THREE.Mesh(geo, colMat.clone());
            mesh.position.copy(p2t(obs.x, obs.y, obs.z_min + height / 2));
            scene.add(mesh);
            obstacleMeshes.push(mesh);

            // Wireframe
            const edges = new THREE.LineSegments(
                new THREE.EdgesGeometry(geo),
                new THREE.LineBasicMaterial({ color: 0x88aacc, transparent: true, opacity: 0.5 })
            );
            edges.position.copy(mesh.position);
            scene.add(edges);
            obstacleMeshes.push(edges);
        }
    }

    // ── Gates ─────────────────────────────────────────────────
    for (const gate of (layout.gates || [])) {
        const frame = _createGateFrame(gate);
        scene.add(frame);
        gateFrames.push(frame);
    }

    // ── Thermal Zones ─────────────────────────────────────────
    for (const tz of (layout.thermals || [])) {
        const emitter = _createThermalEmitter(tz);
        scene.add(emitter);
        thermalEmitters.push(emitter);
    }

    // ── EW Zones ──────────────────────────────────────────────
    for (const zone of (layout.ew_zones?.gps_denial || [])) {
        const dome = _createEWDome(zone, 0xff2222);
        scene.add(dome);
        ewDomes.push(dome);
    }
    for (const zone of (layout.ew_zones?.jamming || [])) {
        const dome = _createEWDome(zone, 0xff8800);
        scene.add(dome);
        ewDomes.push(dome);
    }

    // ── Gun Turrets ────────────────────────────────────────────
    for (const t of (layout.turrets || [])) {
        const turret = _createTurretMesh(t);
        scene.add(turret.group);
        turretMeshes.push(turret);
    }
}

function _createGateFrame(gate) {
    const w = gate.width;
    const h = gate.height;
    const depth = 0.15;

    // Create a rectangular frame (hollow box wireframe)
    const geo = new THREE.BoxGeometry(depth, h, w);
    const edges = new THREE.EdgesGeometry(geo);
    const mat = new THREE.LineBasicMaterial({
        color: GATE_GRAY.getHex(), transparent: true, opacity: 0.7, linewidth: 2,
    });
    const frame = new THREE.LineSegments(edges, mat);

    // Also add a semi-transparent fill for the gate opening
    const fillGeo = new THREE.PlaneGeometry(w, h);
    const fillMat = new THREE.MeshBasicMaterial({
        color: GATE_GREEN.getHex(), transparent: true, opacity: 0.05,
        side: THREE.DoubleSide,
    });
    const fill = new THREE.Mesh(fillGeo, fillMat);
    // Rotate plane to face along gate normal (default: facing X axis)
    fill.rotation.y = Math.PI / 2;
    frame.add(fill);
    frame.userData.fill = fill;

    // Position: physics→three.js coord swap
    frame.position.copy(p2t(gate.x, gate.y, gate.z));

    return frame;
}

function _createThermalEmitter(tz) {
    // Orange particle column for thermal zones
    const count = 60;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const r = Math.random() * tz.radius;
        const angle = Math.random() * Math.PI * 2;
        positions[i * 3] = tz.x + r * Math.cos(angle);
        positions[i * 3 + 1] = Math.random() * 15;  // Y (up in Three.js)
        positions[i * 3 + 2] = tz.y + r * Math.sin(angle);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
        color: THERMAL_COLOR.getHex(), size: 0.15,
        transparent: true, opacity: 0.3,
        blending: THREE.AdditiveBlending,
    });
    const particles = new THREE.Points(geo, mat);
    particles.userData.zone = tz;
    return particles;
}

function _createEWDome(zone, color) {
    // Semi-transparent dome showing EW coverage
    const geo = new THREE.SphereGeometry(zone.radius, 24, 16, 0, Math.PI * 2, 0, Math.PI / 2);
    const mat = new THREE.MeshBasicMaterial({
        color: color, transparent: true, opacity: 0.06,
        side: THREE.DoubleSide, depthWrite: false,
    });
    const dome = new THREE.Mesh(geo, mat);
    dome.position.copy(p2t(zone.x, zone.y, 0));
    return dome;
}

// ══════════════════════════════════════════════════════════════
//  Gun Turret Construction
// ══════════════════════════════════════════════════════════════

function _createTurretMesh(turretData) {
    const group = new THREE.Group();
    const pos = p2t(turretData.x, turretData.y, turretData.z);
    group.position.copy(pos);
    group.userData.id = turretData.id;

    // ── Pedestal (dark octagonal column) ─────────────────────
    const pedestalGeo = new THREE.CylinderGeometry(0.3, 0.4, 1.0, 8);
    const pedestalMat = new THREE.MeshStandardMaterial({
        color: 0x334455, metalness: 0.7, roughness: 0.3,
    });
    const pedestal = new THREE.Mesh(pedestalGeo, pedestalMat);
    pedestal.position.y = 0.5;  // half-height above ground
    group.add(pedestal);

    // ── Turret head (rotating part) ─────────────────────────
    const headGroup = new THREE.Group();
    headGroup.position.y = 1.0;  // on top of pedestal

    // Housing — squat box
    const housingGeo = new THREE.BoxGeometry(0.5, 0.3, 0.5);
    const housingMat = new THREE.MeshStandardMaterial({
        color: 0x556677, metalness: 0.6, roughness: 0.4,
    });
    const housing = new THREE.Mesh(housingGeo, housingMat);
    headGroup.add(housing);

    // Barrel — long cylinder pointing forward (+Z in Three.js = +Y physics)
    const barrelGeo = new THREE.CylinderGeometry(0.06, 0.06, 1.2, 8);
    barrelGeo.rotateX(Math.PI / 2);  // point along Z
    const barrelMat = new THREE.MeshStandardMaterial({
        color: 0x667788, metalness: 0.8, roughness: 0.2,
    });
    const barrel = new THREE.Mesh(barrelGeo, barrelMat);
    barrel.position.z = 0.6;  // extend forward from housing
    barrel.position.y = 0.05;
    headGroup.add(barrel);

    // Muzzle flash (initially invisible)
    const flashGeo = new THREE.SphereGeometry(0.25, 8, 8);
    const flashMat = new THREE.MeshBasicMaterial({
        color: 0xff6600, transparent: true, opacity: 0.0,
        blending: THREE.AdditiveBlending,
    });
    const flash = new THREE.Mesh(flashGeo, flashMat);
    flash.position.z = 1.2;
    flash.position.y = 0.05;
    headGroup.add(flash);

    // Point light for flash
    const flashLight = new THREE.PointLight(0xff6600, 0, 8);
    flashLight.position.copy(flash.position);
    headGroup.add(flashLight);

    // Wireframe outline for visibility
    const wireGeo = new THREE.EdgesGeometry(housingGeo);
    const wireMat = new THREE.LineBasicMaterial({
        color: 0x88aacc, transparent: true, opacity: 0.5,
    });
    const wireframe = new THREE.LineSegments(wireGeo, wireMat);
    headGroup.add(wireframe);

    group.add(headGroup);

    return {
        group,
        headGroup,
        flash,
        flashLight,
        id: turretData.id,
    };
}

function _updateTurrets(turretStates, animTime) {
    if (!turretStates || !turretMeshes.length) return;

    for (const state of turretStates) {
        const turret = turretMeshes.find(t => t.id === state.id);
        if (!turret) continue;

        // Aim barrel at drone: state.ax/ay/az is direction in physics coords
        // Convert to Three.js coords: (ax, az, ay)
        const aimDir = new THREE.Vector3(state.ax, state.az, state.ay).normalize();
        const aimTarget = turret.group.position.clone().add(aimDir.multiplyScalar(5));
        turret.headGroup.lookAt(aimTarget);

        // Muzzle flash
        if (state.flash) {
            turret.flash.material.opacity = 0.9;
            turret.flashLight.intensity = 3.0;
            turret.flash.scale.setScalar(1.0 + Math.random() * 0.5);
        } else {
            // Fade out
            turret.flash.material.opacity *= 0.85;
            turret.flashLight.intensity *= 0.85;
            if (turret.flash.material.opacity < 0.01) {
                turret.flash.material.opacity = 0;
                turret.flashLight.intensity = 0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════
//  Projectile Management
// ══════════════════════════════════════════════════════════════

function _updateProjectiles(projectiles) {
    while (projectileMeshes.length < projectiles.length) {
        const geo = new THREE.SphereGeometry(0.12, 8, 8);
        const mat = new THREE.MeshStandardMaterial({
            color: PROJ_RED.getHex(), emissive: PROJ_RED.getHex(),
            emissiveIntensity: 0.8,
        });
        const mesh = new THREE.Mesh(geo, mat);

        // Trail line
        const trailGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0),
        ]);
        const trailMat = new THREE.LineBasicMaterial({
            color: PROJ_RED.getHex(), transparent: true, opacity: 0.5,
        });
        const trail = new THREE.Line(trailGeo, trailMat);
        scene.add(trail);
        mesh.userData.trail = trail;

        scene.add(mesh);
        projectileMeshes.push(mesh);
    }

    for (let i = 0; i < projectileMeshes.length; i++) {
        const mesh = projectileMeshes[i];
        if (i < projectiles.length) {
            const p = projectiles[i];
            mesh.position.copy(p2t(p.x, p.y, p.z));
            mesh.visible = true;

            // Update trail
            if (mesh.userData.trail) {
                const trail = mesh.userData.trail;
                const pos = trail.geometry.attributes.position.array;
                const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz) || 1;
                const len = 1.5;
                const tailPos = p2t(
                    p.x - p.vx / speed * len,
                    p.y - p.vy / speed * len,
                    p.z - p.vz / speed * len
                );
                pos[0] = mesh.position.x; pos[1] = mesh.position.y; pos[2] = mesh.position.z;
                pos[3] = tailPos.x; pos[4] = tailPos.y; pos[5] = tailPos.z;
                trail.geometry.attributes.position.needsUpdate = true;
                trail.visible = true;
            }
        } else {
            mesh.visible = false;
            if (mesh.userData.trail) mesh.userData.trail.visible = false;
        }
    }
}

// ══════════════════════════════════════════════════════════════
//  Update (called from app.js on each WS message)
// ══════════════════════════════════════════════════════════════

function update(msg) {
    latestMsg = msg;
    if (msg.drone) {
        targetDrone = msg.drone;
    }
    // Swarm data (all N envs' positions)
    if (msg.swarm) {
        if (!_swarmBuilt) _initSwarm(msg.swarm.length);
        swarmTargets = msg.swarm;
    }
    // Update task target marker for non-gate modes
    _updateTaskTarget(msg.task_target);
}

function _updateTaskTarget(target) {
    if (!scene) return;

    // Remove old marker if type changed, target cleared, or stage changed (hover box resize)
    const needsRebuild = taskTargetMarker && (
        !target ||
        taskTargetMarker.userData.type !== target.type ||
        (target.type === 'hover' && taskTargetMarker.userData.stage !== target.stage) ||
        (target.type === 'altitude_change' && taskTargetMarker.userData.cylCount !== (target.cylinders || []).length)
    );
    if (needsRebuild) {
        scene.remove(taskTargetMarker);
        taskTargetMarker = null;
    }

    if (!target) return;

    if (target.type === 'hover' || target.type === 'takeoff') {
        // Translucent cylinder showing the hover zone
        if (!taskTargetMarker) {
            const r = target.radius;
            const altLo = target.alt_lo;
            const altHi = target.alt_hi;
            const height = altHi - altLo;
            const altMid = (altLo + altHi) / 2.0;

            // Wireframe cylinder for the zone boundary
            const cylGeo = new THREE.CylinderGeometry(r, r, height, 24, 1, true);
            const cylMat = new THREE.MeshBasicMaterial({
                color: 0x00ff88, transparent: true, opacity: 0.08,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
                depthWrite: false,
            });
            const cyl = new THREE.Mesh(cylGeo, cylMat);

            // Wireframe edges
            const edgeGeo = new THREE.EdgesGeometry(
                new THREE.CylinderGeometry(r, r, height, 24, 1, true)
            );
            const edgeMat = new THREE.LineBasicMaterial({
                color: 0x00ff88, transparent: true, opacity: 0.35,
            });
            const edges = new THREE.LineSegments(edgeGeo, edgeMat);
            cyl.add(edges);

            // Top and bottom rings (more visible)
            const ringGeo = new THREE.RingGeometry(r * 0.95, r, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0x00ff88, transparent: true, opacity: 0.15,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
            });
            const topRing = new THREE.Mesh(ringGeo, ringMat);
            topRing.rotation.x = -Math.PI / 2;
            topRing.position.y = height / 2;
            cyl.add(topRing);
            const botRing = topRing.clone();
            botRing.position.y = -height / 2;
            cyl.add(botRing);

            // Position: physics→three.js
            cyl.position.copy(p2t(target.cx, target.cy, altMid));

            taskTargetMarker = cyl;
            taskTargetMarker.userData.type = target.type;
            taskTargetMarker.userData.stage = target.stage || 0;
            scene.add(taskTargetMarker);
        }
        // Pulse opacity — brighter when hold timer is filling
        const holdFrac = target.hold_time / target.hold_required;
        const baseOp = 0.06 + holdFrac * 0.12;
        taskTargetMarker.material.opacity = baseOp + 0.04 * Math.sin(animTime * 3);
    } else if (target.type === 'fly_to_point') {
        if (!taskTargetMarker) {
            const geo = new THREE.SphereGeometry(0.5, 16, 16);
            const mat = new THREE.MeshBasicMaterial({
                color: 0x00ff88, transparent: true, opacity: 0.4,
                blending: THREE.AdditiveBlending,
            });
            taskTargetMarker = new THREE.Mesh(geo, mat);
            const ringGeo = new THREE.RingGeometry(target.radius - 0.1, target.radius, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0x00ff88, transparent: true, opacity: 0.2,
                side: THREE.DoubleSide,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = -Math.PI / 2;
            taskTargetMarker.add(ring);
            taskTargetMarker.userData.type = 'fly_to_point';
            scene.add(taskTargetMarker);
        }
        taskTargetMarker.position.copy(p2t(target.x, target.y, target.z));
        taskTargetMarker.material.opacity = 0.25 + 0.15 * Math.sin(animTime * 3);
    } else if (target.type === 'altitude_change' && target.cylinders) {
        if (!taskTargetMarker) {
            // Build group with 3 cylinders
            taskTargetMarker = new THREE.Group();
            taskTargetMarker.userData.type = 'altitude_change';
            taskTargetMarker.userData.cylCount = target.cylinders.length;
            taskTargetMarker.userData.cyls = [];

            for (let c = 0; c < target.cylinders.length; c++) {
                const cyl = target.cylinders[c];
                const r = cyl.radius;
                const h = cyl.height * 2;  // half-height → full height

                // Translucent cylinder body
                const cylGeo = new THREE.CylinderGeometry(r, r, h, 24, 1, true);
                const cylMat = new THREE.MeshBasicMaterial({
                    color: 0xffaa00, transparent: true, opacity: 0.06,
                    side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
                    depthWrite: false,
                });
                const cylMesh = new THREE.Mesh(cylGeo, cylMat);

                // Wireframe edges
                const edgeGeo = new THREE.EdgesGeometry(
                    new THREE.CylinderGeometry(r, r, h, 24, 1, true)
                );
                const edgeMat = new THREE.LineBasicMaterial({
                    color: 0xffaa00, transparent: true, opacity: 0.3,
                });
                cylMesh.add(new THREE.LineSegments(edgeGeo, edgeMat));

                // Top and bottom rings
                const ringGeo = new THREE.RingGeometry(r * 0.9, r, 32);
                const ringMat = new THREE.MeshBasicMaterial({
                    color: 0xffaa00, transparent: true, opacity: 0.12,
                    side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
                });
                const topRing = new THREE.Mesh(ringGeo, ringMat);
                topRing.rotation.x = -Math.PI / 2;
                topRing.position.y = h / 2;
                cylMesh.add(topRing);
                const botRing = topRing.clone();
                botRing.position.y = -h / 2;
                cylMesh.add(botRing);

                // Number label (1, 2, 3) — simple sprite
                const canvas = document.createElement('canvas');
                canvas.width = 64; canvas.height = 64;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#ffaa00';
                ctx.font = 'bold 48px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(String(c + 1), 32, 32);
                const tex = new THREE.CanvasTexture(canvas);
                const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.7 });
                const sprite = new THREE.Sprite(spriteMat);
                sprite.scale.set(1.2, 1.2, 1);
                sprite.position.y = h / 2 + 0.8;
                cylMesh.add(sprite);

                cylMesh.position.copy(p2t(cyl.x, cyl.y, cyl.z));
                taskTargetMarker.add(cylMesh);
                taskTargetMarker.userData.cyls.push({ mesh: cylMesh, mat: cylMat, edgeMat: edgeMat });
            }
            scene.add(taskTargetMarker);
        }
        // Update positions (they change on episode reset due to shuffle)
        const cyls = taskTargetMarker.userData.cyls;
        for (let c = 0; c < target.cylinders.length && c < cyls.length; c++) {
            const cyl = target.cylinders[c];
            cyls[c].mesh.position.copy(p2t(cyl.x, cyl.y, cyl.z));
        }
        // Update colors: completed=green, active=orange pulse, pending=dim grey
        for (let c = 0; c < cyls.length; c++) {
            const { mat, edgeMat } = cyls[c];
            if (c < target.changes_done) {
                // Completed — green, solid
                mat.color.setHex(0x00ff88);
                mat.opacity = 0.10;
                edgeMat.color.setHex(0x00ff88);
                edgeMat.opacity = 0.5;
            } else if (c === target.current_cyl) {
                // Active — orange, pulsing
                const holdFrac = target.hold_time / target.hold_required;
                mat.color.setHex(0xffaa00);
                mat.opacity = 0.08 + holdFrac * 0.12 + 0.04 * Math.sin(animTime * 4);
                edgeMat.color.setHex(0xffaa00);
                edgeMat.opacity = 0.4 + holdFrac * 0.4;
            } else {
                // Pending — dim
                mat.color.setHex(0x666666);
                mat.opacity = 0.03;
                edgeMat.color.setHex(0x666666);
                edgeMat.opacity = 0.15;
            }
        }
    } else if (target.type === 'land') {
        // ── Landing zone: concentric rings on the ground (helicopter pad) ──
        if (!taskTargetMarker) {
            taskTargetMarker = new THREE.Group();
            taskTargetMarker.userData.type = 'land';
            const r = target.radius || 2.0;

            // Outer ring
            const outerGeo = new THREE.RingGeometry(r - 0.08, r + 0.08, 48);
            const outerMat = new THREE.MeshBasicMaterial({
                color: 0xff4466, transparent: true, opacity: 0.25,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const outer = new THREE.Mesh(outerGeo, outerMat);
            outer.rotation.x = -Math.PI / 2;
            outer.position.y = 0.04;
            taskTargetMarker.add(outer);

            // Middle ring
            const midGeo = new THREE.RingGeometry(r * 0.55 - 0.06, r * 0.55 + 0.06, 36);
            const midMat = new THREE.MeshBasicMaterial({
                color: 0xff4466, transparent: true, opacity: 0.18,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const mid = new THREE.Mesh(midGeo, midMat);
            mid.rotation.x = -Math.PI / 2;
            mid.position.y = 0.04;
            taskTargetMarker.add(mid);

            // Center dot
            const dotGeo = new THREE.CircleGeometry(0.25, 16);
            const dotMat = new THREE.MeshBasicMaterial({
                color: 0xff4466, transparent: true, opacity: 0.3,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const dot = new THREE.Mesh(dotGeo, dotMat);
            dot.rotation.x = -Math.PI / 2;
            dot.position.y = 0.05;
            taskTargetMarker.add(dot);

            // Cross-hairs (two perpendicular lines through center)
            for (let a = 0; a < 2; a++) {
                const pts = [
                    new THREE.Vector3(-r * 0.35, 0.04, 0),
                    new THREE.Vector3(r * 0.35, 0.04, 0),
                ];
                const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
                const lineMat = new THREE.LineBasicMaterial({
                    color: 0xff4466, transparent: true, opacity: 0.2,
                });
                const line = new THREE.Line(lineGeo, lineMat);
                line.rotation.y = a * Math.PI / 2;
                taskTargetMarker.add(line);
            }

            taskTargetMarker.userData.outerMat = outerMat;
            taskTargetMarker.userData.dotMat = dotMat;
            taskTargetMarker.position.copy(p2t(target.cx, target.cy, 0));
            scene.add(taskTargetMarker);
        }
        // Gentle pulse animation
        const { outerMat, dotMat } = taskTargetMarker.userData;
        outerMat.opacity = 0.20 + 0.08 * Math.sin(animTime * 2);
        dotMat.opacity = 0.25 + 0.10 * Math.sin(animTime * 2);

    } else if (target.type === 'yaw') {
        // ── Yaw beacon: pillar at target direction + compass ring + progress dots ──
        const BEACON_DIST = 6.0;
        const PILLAR_H = 6.0;

        if (!taskTargetMarker) {
            taskTargetMarker = new THREE.Group();
            taskTargetMarker.userData.type = 'yaw';

            // Position group at center (ground level)
            taskTargetMarker.position.copy(p2t(target.cx, target.cy, 0));

            // Containment zone cylinder — the drone must stay inside this
            const zr = target.zone_radius || 3.0;
            const zAltLo = target.zone_alt_lo || 2.0;
            const zAltHi = target.zone_alt_hi || 8.0;
            const zHeight = zAltHi - zAltLo;
            const zAltMid = (zAltLo + zAltHi) / 2.0;

            const zoneGeo = new THREE.CylinderGeometry(zr, zr, zHeight, 24, 1, true);
            const zoneMat = new THREE.MeshBasicMaterial({
                color: 0x00aaff, transparent: true, opacity: 0.04,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const zoneMesh = new THREE.Mesh(zoneGeo, zoneMat);
            // Position at altitude midpoint (group is at ground, Three.js Y = physics Z)
            zoneMesh.position.y = zAltMid;
            taskTargetMarker.add(zoneMesh);

            // Zone wireframe edges
            const zEdgeGeo = new THREE.EdgesGeometry(
                new THREE.CylinderGeometry(zr, zr, zHeight, 24, 1, true)
            );
            const zEdgeMat = new THREE.LineBasicMaterial({
                color: 0x00aaff, transparent: true, opacity: 0.15,
            });
            const zEdges = new THREE.LineSegments(zEdgeGeo, zEdgeMat);
            zoneMesh.add(zEdges);

            // Zone top and bottom rings
            const zRingGeo = new THREE.RingGeometry(zr * 0.95, zr, 32);
            const zRingMat = new THREE.MeshBasicMaterial({
                color: 0x00aaff, transparent: true, opacity: 0.08,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
            });
            const zTopRing = new THREE.Mesh(zRingGeo, zRingMat);
            zTopRing.rotation.x = -Math.PI / 2;
            zTopRing.position.y = zHeight / 2;
            zoneMesh.add(zTopRing);
            const zBotRing = zTopRing.clone();
            zBotRing.position.y = -zHeight / 2;
            zoneMesh.add(zBotRing);

            // Static compass ring on ground (at beacon distance)
            const ringGeo = new THREE.RingGeometry(BEACON_DIST - 0.04, BEACON_DIST + 0.04, 64);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0x445566, transparent: true, opacity: 0.07,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = 0.03;
            taskTargetMarker.add(ring);

            // Rotatable sub-group for directional elements
            const dirGroup = new THREE.Group();
            taskTargetMarker.userData.dirGroup = dirGroup;
            taskTargetMarker.add(dirGroup);

            // Beacon pillar at +X (before rotation), base at ground
            const pillarGeo = new THREE.CylinderGeometry(0.15, 0.22, PILLAR_H, 8);
            const pillarMat = new THREE.MeshBasicMaterial({
                color: 0xffaa00, transparent: true, opacity: 0.25,
                blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const pillar = new THREE.Mesh(pillarGeo, pillarMat);
            pillar.position.set(BEACON_DIST, PILLAR_H / 2, 0);
            dirGroup.add(pillar);
            taskTargetMarker.userData.pillarMat = pillarMat;

            // Beacon top glow sphere
            const topGeo = new THREE.SphereGeometry(0.4, 12, 12);
            const topMat = new THREE.MeshBasicMaterial({
                color: 0xffaa00, transparent: true, opacity: 0.5,
                blending: THREE.AdditiveBlending,
            });
            const topGlow = new THREE.Mesh(topGeo, topMat);
            topGlow.position.set(BEACON_DIST, PILLAR_H + 0.3, 0);
            dirGroup.add(topGlow);
            taskTargetMarker.userData.topMat = topMat;

            // Direction line from center to beacon base
            const lineGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0.05, 0),
                new THREE.Vector3(BEACON_DIST, 0.05, 0),
            ]);
            const lineMat = new THREE.LineBasicMaterial({
                color: 0xffaa00, transparent: true, opacity: 0.12,
            });
            dirGroup.add(new THREE.Line(lineGeo, lineMat));
            taskTargetMarker.userData.lineMat = lineMat;

            // Tolerance arc on ground at beacon distance
            const tol = target.tolerance || 0.15;
            const arcGeo = new THREE.RingGeometry(
                BEACON_DIST - 0.35, BEACON_DIST + 0.35, 16, 1,
                -tol, tol * 2
            );
            const arcMat = new THREE.MeshBasicMaterial({
                color: 0xffaa00, transparent: true, opacity: 0.10,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const arc = new THREE.Mesh(arcGeo, arcMat);
            arc.rotation.x = -Math.PI / 2;
            arc.position.y = 0.04;
            dirGroup.add(arc);
            taskTargetMarker.userData.arcMat = arcMat;

            // Progress dots (3 small spheres above the beacon)
            taskTargetMarker.userData.dots = [];
            for (let d = 0; d < 3; d++) {
                const dGeo = new THREE.SphereGeometry(0.18, 8, 8);
                const dMat = new THREE.MeshBasicMaterial({
                    color: 0x666666, transparent: true, opacity: 0.4,
                });
                const dMesh = new THREE.Mesh(dGeo, dMat);
                dMesh.position.set(BEACON_DIST, PILLAR_H + 1.2 + d * 0.5, 0);
                dirGroup.add(dMesh);
                taskTargetMarker.userData.dots.push({ mesh: dMesh, mat: dMat });
            }

            scene.add(taskTargetMarker);
        }

        // Rotate directional group to match target yaw
        // Physics yaw → Three.js Y rotation: negate because p2t flips Y→-Z
        taskTargetMarker.userData.dirGroup.rotation.y = -target.target_yaw;

        // Hold animation — brighten as drone holds correct heading
        const holdFrac = target.hold_time / target.hold_required;
        const { pillarMat, topMat, lineMat, arcMat } = taskTargetMarker.userData;
        pillarMat.opacity = 0.15 + holdFrac * 0.35 + 0.05 * Math.sin(animTime * 4);
        topMat.opacity = 0.30 + holdFrac * 0.50;
        lineMat.opacity = 0.10 + holdFrac * 0.25;
        arcMat.opacity = 0.08 + holdFrac * 0.20;

        // Progress dots: green=done, orange=active, grey=pending
        const dots = taskTargetMarker.userData.dots;
        for (let d = 0; d < 3; d++) {
            if (d < target.changes_done) {
                dots[d].mat.color.setHex(0x00ff88);
                dots[d].mat.opacity = 0.7;
            } else if (d === target.changes_done) {
                dots[d].mat.color.setHex(0xffaa00);
                dots[d].mat.opacity = 0.5 + holdFrac * 0.3;
            } else {
                dots[d].mat.color.setHex(0x666666);
                dots[d].mat.opacity = 0.3;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════
//  Animation Loop
// ══════════════════════════════════════════════════════════════

function _animate() {
    requestAnimationFrame(_animate);
    animTime += 0.016;  // ~60fps

    // ── Drone interpolation ───────────────────────────────────
    if (targetDrone) {
        currentDrone.x += (targetDrone.x - currentDrone.x) * LERP_SPEED;
        currentDrone.y += (targetDrone.y - currentDrone.y) * LERP_SPEED;
        currentDrone.z += (targetDrone.z - currentDrone.z) * LERP_SPEED;

        // Physics→Three.js: (x, z, y)
        droneGroup.position.set(currentDrone.x, currentDrone.z, currentDrone.y);

        // Quaternion: physics [w,x,y,z] → Three.js with Y/Z swap + handedness fix
        // The Y↔Z position swap flips handedness (det=-1), so we negate the
        // vector part after swapping to preserve correct rotation sense.
        const q = new THREE.Quaternion(
            -targetDrone.qx, -targetDrone.qz, -targetDrone.qy, targetDrone.qw
        );
        droneGroup.quaternion.slerp(q, LERP_SPEED);

        // Ground clamp: prevent visual mesh from clipping below y=0
        // Transform arm tip positions by current rotation to find lowest world-Y
        let minY = droneGroup.position.y;
        for (let i = 0; i < ARM_TIP_LOCALS.length; i++) {
            const worldPt = ARM_TIP_LOCALS[i].clone().applyQuaternion(droneGroup.quaternion);
            const wy = droneGroup.position.y + worldPt.y;
            if (wy < minY) minY = wy;
        }
        if (minY < 0) {
            droneGroup.position.y -= minY;  // lift by the amount of penetration
        }

        // Spin propeller blades based on RPM
        if (targetDrone.motor_rpms) {
            for (let i = 0; i < 4; i++) {
                const rpm = targetDrone.motor_rpms[i] || 0;
                const spinSpeed = (rpm / MAX_RPM) * 0.8;
                const dir = (i === 0 || i === 2) ? 1 : -1;
                propGroups[i].rotation.y += spinSpeed * dir;
                guardRings[i].material.opacity = 0.4 + (rpm / MAX_RPM) * 0.4;
            }
        }

        // Shadow
        if (droneShadow) {
            droneShadow.position.set(currentDrone.x, 0.02, currentDrone.y);
            const shadowScale = Math.max(0.2, 1.0 - currentDrone.z / COURSE_H);
            droneShadow.scale.set(shadowScale, shadowScale, 1);
            droneShadow.material.opacity = 0.3 * shadowScale;
        }

        // Camera follow
        if (_chaseMode) {
            // Chase camera: use the Three.js quaternion (already correctly transformed)
            // to get the drone's forward direction, then position camera behind it.
            // This avoids physics↔Three.js sign/convention errors entirely.
            const forward = new THREE.Vector3(1, 0, 0);  // local +X = physics forward
            forward.applyQuaternion(droneGroup.quaternion);
            forward.y = 0;  // project to horizontal plane
            if (forward.length() > 0.01) forward.normalize();
            else forward.set(1, 0, 0);  // fallback if looking straight up/down

            const dronePos = droneGroup.position;
            const camTarget = new THREE.Vector3().copy(dronePos);
            camTarget.addScaledVector(forward, -CHASE_DIST);  // behind
            camTarget.y += CHASE_HEIGHT;                       // above

            camera.position.lerp(camTarget, CHASE_LERP);
            controls.target.lerp(dronePos.clone(), CHASE_LERP * 2);
        } else {
            controls.target.lerp(
                new THREE.Vector3(currentDrone.x, currentDrone.z, currentDrone.y), 0.05
            );
        }
    }

    // ── Swarm ghost drones ────────────────────────────────────
    if (_swarmVisible && swarmTargets.length > 0) {
        for (let i = 0; i < swarmDrones.length; i++) {
            const envIdx = i + 1;  // skip env[0] (primary drone)
            const t = swarmTargets[envIdx];
            if (!t) continue;
            const c = swarmCurrents[i];
            // Lerp position
            c.x += (t[0] - c.x) * 0.2;
            c.y += (t[1] - c.y) * 0.2;
            c.z += (t[2] - c.z) * 0.2;
            swarmDrones[i].position.set(c.x, c.z, c.y);  // physics→Three.js
            // Slerp quaternion (same convention as primary drone)
            const q = new THREE.Quaternion(-t[4], -t[6], -t[5], t[3]);
            swarmDrones[i].quaternion.slerp(q, 0.2);
            // Ground clamp (simple — just prevent Y < 0)
            if (swarmDrones[i].position.y < 0) swarmDrones[i].position.y = 0;
        }
    }

    // ── LIDAR ─────────────────────────────────────────────────
    if (latestMsg?.lidar) {
        _updateLidar(latestMsg.lidar);
    }

    // ── Projectiles ───────────────────────────────────────────
    if (latestMsg?.projectiles) {
        _updateProjectiles(latestMsg.projectiles);
    }

    // ── Gun turrets (aim + flash) ────────────────────────────
    if (latestMsg?.turrets) {
        _updateTurrets(latestMsg.turrets, animTime);
    }

    // ── Adversary drone ──────────────────────────────────────
    if (latestMsg?.adversary) {
        _updateAdversary(latestMsg.adversary);
    } else if (adversaryGroup) {
        adversaryGroup.visible = false;
        if (adversaryTrail) adversaryTrail.visible = false;
    }

    // ── Turret marker pulse ──────────────────────────────────
    if (turretMarker && turretMarker.visible) {
        turretMarker.material.opacity = 0.4 + Math.sin(animTime * 3) * 0.2;
        turretMarker.rotation.y += 0.01;
    }

    // ── Gate coloring ─────────────────────────────────────────
    if (latestMsg && gateFrames.length > 0) {
        const gatesPassed = latestMsg.gates_passed || 0;
        const currentGate = latestMsg.waypoint?.index ?? gatesPassed;

        for (let i = 0; i < gateFrames.length; i++) {
            const frame = gateFrames[i];
            const mat = frame.material;
            const fill = frame.userData.fill;
            if (i < gatesPassed) {
                // Passed — gold, dimmer
                mat.color.set(GATE_GOLD);
                mat.opacity = 0.4;
                if (fill) { fill.material.color.set(GATE_GOLD.getHex()); fill.material.opacity = 0.03; }
            } else if (i === currentGate) {
                // Current target — bright green, pulsing
                const pulse = 0.6 + Math.sin(animTime * 4) * 0.3;
                mat.color.set(GATE_GREEN);
                mat.opacity = pulse;
                if (fill) { fill.material.color.set(GATE_GREEN.getHex()); fill.material.opacity = 0.08; }
            } else {
                // Future — gray
                mat.color.set(GATE_GRAY);
                mat.opacity = 0.3;
                if (fill) { fill.material.color.set(GATE_GRAY.getHex()); fill.material.opacity = 0.02; }
            }
        }
    }

    // ── Moving obstacles animation ────────────────────────────
    if (latestMsg) {
        const simTime = (latestMsg.step || 0) * 0.02;  // RL_DT = 0.02
        for (const { mesh, edges, data } of movingObstacles) {
            const offset = data.amplitude * Math.sin(2 * Math.PI * simTime / data.period);
            const mn = [...data.base_min];
            const mx = [...data.base_max];
            mn[data.axis] += offset;
            mx[data.axis] += offset;
            const sx = mx[0] - mn[0];
            const sy = mx[1] - mn[1];
            const sz = mx[2] - mn[2];
            const pos = p2t(mn[0] + sx / 2, mn[1] + sy / 2, mn[2] + sz / 2);
            mesh.position.copy(pos);
            edges.position.copy(pos);
        }
    }

    // ── Thermal particles animation ───────────────────────────
    for (const emitter of thermalEmitters) {
        const positions = emitter.geometry.attributes.position.array;
        const tz = emitter.userData.zone;
        for (let i = 0; i < positions.length; i += 3) {
            positions[i + 1] += 0.03 + Math.random() * 0.02;  // rise
            if (positions[i + 1] > 15) {
                // Reset to ground
                const r = Math.random() * tz.radius;
                const a = Math.random() * Math.PI * 2;
                positions[i] = tz.x + r * Math.cos(a);
                positions[i + 1] = 0;
                positions[i + 2] = tz.y + r * Math.sin(a);
            }
        }
        emitter.geometry.attributes.position.needsUpdate = true;
    }

    // ── Wind particles ────────────────────────────────────────
    if (latestMsg?.wind && windParticles) {
        const windVec = latestMsg.wind.base;
        const windMag = Math.sqrt(windVec[0] ** 2 + windVec[1] ** 2 + windVec[2] ** 2);
        windParticles.visible = windMag > 0.5;

        if (windParticles.visible) {
            const positions = windParticles.geometry.attributes.position.array;
            for (let i = 0; i < positions.length; i += 3) {
                // Physics→Three.js wind mapping
                positions[i] += windVec[0] * 0.02;
                positions[i + 1] += windVec[2] * 0.02;
                positions[i + 2] += windVec[1] * 0.02;

                // Wrap into course bounds
                if (positions[i] > COURSE_W) positions[i] = 0;
                if (positions[i] < 0) positions[i] = COURSE_W;
                if (positions[i + 1] > COURSE_H) positions[i + 1] = 0;
                if (positions[i + 1] < 0) positions[i + 1] = COURSE_H;
                if (positions[i + 2] > COURSE_D) positions[i + 2] = 0;
                if (positions[i + 2] < 0) positions[i + 2] = COURSE_D;
            }
            windParticles.geometry.attributes.position.needsUpdate = true;
            windParticles.material.opacity = Math.min(0.4, windMag * 0.05);
        }
    }

    controls.update();
    composer.render();
}

// ── LIDAR ray update ──────────────────────────────────────────
function _updateLidar(distances) {
    if (!droneGroup) return;

    const dronePos = droneGroup.position;
    const droneQuat = droneGroup.quaternion;

    for (let i = 0; i < 12; i++) {
        if (i >= lidarLines.length) break;
        const line = lidarLines[i];

        // Body-frame dir → swap Y/Z for Three.js
        const dir = new THREE.Vector3(LIDAR_DIRS[i][0], LIDAR_DIRS[i][2], LIDAR_DIRS[i][1]);
        dir.applyQuaternion(droneQuat).normalize();

        const dist = Math.min(distances[i] || MAX_LIDAR_RANGE, MAX_LIDAR_RANGE);
        const end = dronePos.clone().add(dir.multiplyScalar(dist));

        const positions = line.geometry.attributes.position.array;
        positions[0] = dronePos.x; positions[1] = dronePos.y; positions[2] = dronePos.z;
        positions[3] = end.x; positions[4] = end.y; positions[5] = end.z;
        line.geometry.attributes.position.needsUpdate = true;

        // Color: green (far) → red (close)
        const t = dist / MAX_LIDAR_RANGE;
        line.material.color.setHSL(t * 0.35, 1, 0.5);
        line.material.opacity = 0.3 + (1 - t) * 0.4;
        line.visible = true;
    }
}

// ── Chase Camera ──────────────────────────────────────────────
function setChaseCamera(enabled) {
    _chaseMode = enabled;
    if (enabled) {
        controls.enableRotate = false;  // lock orbit during chase
    } else {
        controls.enableRotate = true;
    }
}

// ── Adversary Drone ───────────────────────────────────────────
function _buildAdversary() {
    if (adversaryGroup) {
        scene.remove(adversaryGroup);
    }
    adversaryGroup = new THREE.Group();
    adversaryGroup.visible = false;

    // Small red drone body (80% scale of primary)
    const bodyGeo = new THREE.SphereGeometry(0.08, 8, 6);
    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0xff2222, emissive: 0x880000, metalness: 0.7, roughness: 0.3,
    });
    adversaryGroup.add(new THREE.Mesh(bodyGeo, bodyMat));

    // Arms (4x, red/dark)
    const armMat = new THREE.MeshStandardMaterial({ color: 0x661111 });
    for (let i = 0; i < 4; i++) {
        const angle = (i * Math.PI / 2) + Math.PI / 4;
        const armGeo = new THREE.CylinderGeometry(0.008, 0.008, 0.1, 4);
        armGeo.rotateZ(Math.PI / 2);
        const arm = new THREE.Mesh(armGeo, armMat);
        arm.position.set(Math.cos(angle) * 0.05, 0, Math.sin(angle) * 0.05);
        arm.lookAt(0, 0, 0);
        adversaryGroup.add(arm);
    }

    // Pulsing red point light
    const light = new THREE.PointLight(0xff0000, 2, 8);
    light.position.set(0, 0.1, 0);
    adversaryGroup.add(light);
    adversaryGroup.userData.light = light;

    scene.add(adversaryGroup);

    // Trail (line segments)
    const trailGeo = new THREE.BufferGeometry();
    const trailPositions = new Float32Array(60 * 3);  // 60 trail points
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    const trailMat = new THREE.LineBasicMaterial({
        color: 0xff2200, transparent: true, opacity: 0.5,
    });
    adversaryTrail = new THREE.Line(trailGeo, trailMat);
    adversaryTrail.visible = false;
    adversaryTrail.userData.points = [];
    scene.add(adversaryTrail);
}

function _updateAdversary(advData) {
    if (!adversaryGroup) _buildAdversary();

    if (!advData || !advData.active) {
        adversaryGroup.visible = false;
        if (adversaryTrail) adversaryTrail.visible = false;
        return;
    }

    adversaryGroup.visible = true;
    const pos = p2t(advData.x, advData.y, advData.z);
    adversaryGroup.position.lerp(pos, 0.3);

    // Point toward velocity direction
    if (Math.abs(advData.vx) + Math.abs(advData.vy) + Math.abs(advData.vz) > 0.1) {
        const vel = p2t(advData.vx, advData.vy, advData.vz);
        const lookAt = adversaryGroup.position.clone().add(vel);
        adversaryGroup.lookAt(lookAt);
    }

    // Pulse the light
    const light = adversaryGroup.userData.light;
    if (light) light.intensity = 1.5 + Math.sin(animTime * 8) * 1.0;

    // Update trail
    if (adversaryTrail) {
        const pts = adversaryTrail.userData.points;
        pts.push(adversaryGroup.position.clone());
        if (pts.length > 60) pts.shift();

        const positions = adversaryTrail.geometry.attributes.position.array;
        for (let i = 0; i < pts.length; i++) {
            positions[i * 3] = pts[i].x;
            positions[i * 3 + 1] = pts[i].y;
            positions[i * 3 + 2] = pts[i].z;
        }
        // Zero out unused slots
        for (let i = pts.length; i < 60; i++) {
            positions[i * 3] = positions[i * 3 + 1] = positions[i * 3 + 2] = 0;
        }
        adversaryTrail.geometry.attributes.position.needsUpdate = true;
        adversaryTrail.geometry.setDrawRange(0, pts.length);
        adversaryTrail.visible = pts.length > 2;
    }
}

// ── Turret Marker ─────────────────────────────────────────────
function _buildTurret() {
    if (turretMarker) return;
    // Wireframe pyramid at fixed turret position (0, COURSE_D/2, 10) in physics coords
    const geo = new THREE.ConeGeometry(0.4, 0.8, 4);
    const mat = new THREE.MeshBasicMaterial({
        color: 0x44aaff, wireframe: true, transparent: true, opacity: 0.6,
    });
    turretMarker = new THREE.Mesh(geo, mat);
    const tPos = p2t(0, COURSE_D / 2, 10);
    turretMarker.position.copy(tPos);
    turretMarker.visible = false;
    scene.add(turretMarker);
}

function setWeaponMode(enabled) {
    _weaponMode = enabled;
    if (turretMarker) turretMarker.visible = enabled;
    if (!turretMarker && enabled) _buildTurret();
}

function getWeaponMode() { return _weaponMode; }

// ── Export ─────────────────────────────────────────────────────
window.Arena = { init, update, buildCourse, setChaseCamera, setWeaponMode, getWeaponMode, setSwarmMode };
