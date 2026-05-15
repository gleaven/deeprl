/**
 * DEEPRL — Three.js 3D Soccer Arena Renderer
 * Decoupled from WebSocket: stores latest state, renders at 60fps.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ── Constants ─────────────────────────────────────────────────
const FIELD_W = 30.0;
const FIELD_H = 20.0;
const GOAL_W = 5.0;
const AGENT_RADIUS = 0.5;
const BALL_RADIUS = 0.18;
const MAX_TRAIL = 12;
const LERP_SPEED = 0.2; // interpolation factor toward target state

// Articulated leg geometry constants — legs must be taller than ball (r=0.18)
const BODY_RAISE = 0.55;         // raise body above ball height
const HIP_HEIGHT = 0.65;         // hip pivots well above ball top (0.70)
const LEG_OFFSET_X = 0.16;       // left/right offset from center

// Thigh segment (hip → knee)
const THIGH_LENGTH = 0.36;       // longer thigh for proper proportions
const THIGH_RADIUS_TOP = 0.065;  // wider at hip
const THIGH_RADIUS_BOT = 0.05;   // narrows toward knee

// Shin segment (knee → ankle)
const SHIN_LENGTH = 0.32;        // longer shin — knee well above ball center
const SHIN_RADIUS_TOP = 0.05;    // matches thigh bottom
const SHIN_RADIUS_BOT = 0.038;   // narrows toward ankle

// Boot / foot
const BOOT_LENGTH = 0.16;        // slightly larger boot
const BOOT_WIDTH = 0.09;         // slightly wider
const BOOT_HEIGHT = 0.06;        // low profile

// Colors
const CYAN = new THREE.Color(0x00e5ff);
const MAGENTA = new THREE.Color(0xff00aa);
const WHITE = new THREE.Color(0xffffff);
const FIELD_GREEN = new THREE.Color(0x0a1a0a);
const GRID_COLOR = new THREE.Color(0x00e5ff);

// ── Module state ──────────────────────────────────────────────
let scene, camera, renderer, composer, controls;
let agentMeshes = [];      // 4 agent groups
let ballMesh;
let trailMeshes = { agents: [[], [], [], []], ball: [] };
let arrowHelpers = [];
let celebrationParticles = null;
let celebrationClock = 0;

// Latest state from WebSocket (set externally)
let targetState = null;
// Current interpolated state — agentMap keyed by agent id
let currentAgentMap = {};   // {id: {x, y, vx, vy, team}}
let targetAgentMap = {};
let currentBall = null;
let targetBall = null;

// Per-agent animation state (persistent across frames)
const agentAnimState = [
    { jukeTimer: 0, jukeSide: 1 },
    { jukeTimer: 0, jukeSide: 1 },
    { jukeTimer: 0, jukeSide: 1 },
    { jukeTimer: 0, jukeSide: 1 },
];

// ── Leg Part Accessors ──────────────────────────────────────
// Hip pivot hierarchy: [0]=thigh mesh, [1]=knee pivot group
//   Knee pivot: [0]=shin mesh, [1]=ankle pivot group
//     Ankle pivot: [0]=boot mesh
function _getKneePivot(hipPivot)  { return hipPivot.children[1]; }
function _getAnklePivot(kneePivot) { return kneePivot.children[1]; }

// ── Procedural Knee IK ──────────────────────────────────────
// Compute knee bend from hip angle. When thigh swings forward,
// shin trails behind (knee bends). Prevents hyper-extension.
function _kneeFromHip(hipAngle, bendFactor = 1.4) {
    const MIN_BEND = 0.05;
    const forwardBend = Math.max(0, hipAngle) * bendFactor;
    const backBend = Math.max(0, -hipAngle) * bendFactor * 0.3;
    return Math.max(MIN_BEND, forwardBend + backBend);
}

// ── Walk Animation ──────────────────────────────────────────
function _animateWalk(leftHip, rightHip, speed, now) {
    const walkSpeed = Math.min(speed, 6);
    const walkPhase = now * 0.006 * walkSpeed;
    const swingAmplitude = Math.min(speed * 0.1, 0.6);

    const lAngle = Math.sin(walkPhase) * swingAmplitude;
    const rAngle = Math.sin(walkPhase + Math.PI) * swingAmplitude;

    leftHip.rotation.x = lAngle;
    rightHip.rotation.x = rAngle;

    const lKnee = _getKneePivot(leftHip);
    const rKnee = _getKneePivot(rightHip);
    lKnee.rotation.x = _kneeFromHip(lAngle);
    rKnee.rotation.x = _kneeFromHip(rAngle);

    // Ankle compensates to keep feet roughly flat
    _getAnklePivot(lKnee).rotation.x = -(lAngle + lKnee.rotation.x) * 0.5;
    _getAnklePivot(rKnee).rotation.x = -(rAngle + rKnee.rotation.x) * 0.5;
}

// ── Dribble Animation ───────────────────────────────────────
// Faster cadence, shorter stride, crouched stance
function _animateDribble(leftHip, rightHip, speed, now) {
    const dribblePhase = now * 0.012 * Math.max(speed, 2);
    const amp = 0.2;

    const lAngle = Math.sin(dribblePhase) * amp;
    const rAngle = Math.sin(dribblePhase + Math.PI) * amp;

    leftHip.rotation.x = lAngle;
    rightHip.rotation.x = rAngle;

    const lKnee = _getKneePivot(leftHip);
    const rKnee = _getKneePivot(rightHip);
    lKnee.rotation.x = _kneeFromHip(lAngle) + 0.15;  // extra bend
    rKnee.rotation.x = _kneeFromHip(rAngle) + 0.15;

    _getAnklePivot(lKnee).rotation.x = -(lAngle + lKnee.rotation.x) * 0.7;
    _getAnklePivot(rKnee).rotation.x = -(rAngle + rKnee.rotation.x) * 0.7;
}

// ── Kick Animation (3-phase: wind-up → strike → follow-through) ──
function _animateKick(leftHip, rightHip, kickLeg, kicking, kickPower) {
    const kickHip = kickLeg === 0 ? leftHip : rightHip;
    const plantHip = kickLeg === 0 ? rightHip : leftHip;
    const kickKnee = _getKneePivot(kickHip);
    const plantKnee = _getKneePivot(plantHip);
    const kickAnkle = _getAnklePivot(kickKnee);
    const plantAnkle = _getAnklePivot(plantKnee);
    const power = Math.max(kickPower, 0.3);

    // Plant leg: stable, slight bend
    plantHip.rotation.x = -0.1;
    plantKnee.rotation.x = 0.2;
    plantAnkle.rotation.x = 0.0;

    if (kicking > 0.7) {
        // Wind-up: thigh pulls back, knee deeply bent
        const t = (kicking - 0.7) / 0.3;
        kickHip.rotation.x = -0.8 * power * t;
        kickKnee.rotation.x = 1.2 * power * t;
        kickAnkle.rotation.x = 0.3 * t;
    } else if (kicking > 0.3) {
        // Strike: explosive forward, knee snaps straight
        const t = (kicking - 0.3) / 0.4;
        const progress = 1.0 - t;
        kickHip.rotation.x = -0.8 * power * (1 - progress * 2.5);
        kickKnee.rotation.x = 1.2 * power * (1 - progress * 1.1);
        kickAnkle.rotation.x = -0.4 * progress;
    } else {
        // Follow-through: leg forward, decelerating
        const t = kicking / 0.3;
        kickHip.rotation.x = 1.2 * power * t;
        kickKnee.rotation.x = 0.1 * t;
        kickAnkle.rotation.x = -0.3 * t;
    }
}

// ── Knee Strike Animation ──────────────────────────────────
// Used when ball is at mid-height — thigh lifts sharply upward
function _animateKneeStrike(leftHip, rightHip, kickLeg, kicking, kickPower) {
    const kickHip = kickLeg === 0 ? leftHip : rightHip;
    const plantHip = kickLeg === 0 ? rightHip : leftHip;
    const kickKnee = _getKneePivot(kickHip);
    const plantKnee = _getKneePivot(plantHip);
    const kickAnkle = _getAnklePivot(kickKnee);
    const plantAnkle = _getAnklePivot(plantKnee);
    const power = Math.max(kickPower, 0.3);

    // Plant leg: stable
    plantHip.rotation.x = -0.05;
    plantKnee.rotation.x = 0.15;
    plantAnkle.rotation.x = 0.0;

    if (kicking > 0.5) {
        // Wind-up: crouch slightly
        const t = (kicking - 0.5) / 0.5;
        kickHip.rotation.x = -0.3 * t;
        kickKnee.rotation.x = 0.6 * t;
        kickAnkle.rotation.x = 0.2 * t;
    } else {
        // Strike: thigh lifts high, shin hangs back (knee strike)
        const t = kicking / 0.5;
        kickHip.rotation.x = 0.9 * power * t;   // thigh rises
        kickKnee.rotation.x = 1.0 * power * t;  // knee bent tight
        kickAnkle.rotation.x = 0.3 * t;
    }
}

// ── Header Animation ───────────────────────────────────────
// Body dips forward, legs stay neutral — ball is at head height
function _animateHeader(body, kicking) {
    // Quick forward lean
    const lean = kicking * 0.25;
    body.position.z = -lean;      // lean forward
    body.position.y += lean * 0.1; // slight upward stretch
}

// ── Jump Tuck Animation ────────────────────────────────────
// Legs tuck underneath body while airborne
function _animateJumpTuck(leftHip, rightHip, agentZ) {
    const tuck = Math.min(agentZ * 2.0, 1.0);  // ramp up tuck with height

    // Both legs pull up and bend at knee
    leftHip.rotation.x = 0.5 * tuck;
    rightHip.rotation.x = 0.5 * tuck;

    const lKnee = _getKneePivot(leftHip);
    const rKnee = _getKneePivot(rightHip);
    lKnee.rotation.x = 0.8 * tuck;  // tight knee bend
    rKnee.rotation.x = 0.8 * tuck;

    _getAnklePivot(lKnee).rotation.x = -0.3 * tuck;
    _getAnklePivot(rKnee).rotation.x = -0.3 * tuck;
}

// ── Juke Animation ──────────────────────────────────────────
// Returns true if juke is active (caller should skip walk/dribble)
function _animateJuke(leftHip, rightHip, jukeTimer, jukeSide) {
    if (jukeTimer < 0.05) return false;
    const intensity = jukeTimer;

    const plantHip = jukeSide > 0 ? leftHip : rightHip;
    const pushHip = jukeSide > 0 ? rightHip : leftHip;
    const plantKnee = _getKneePivot(plantHip);
    const pushKnee = _getKneePivot(pushHip);
    const plantAnkle = _getAnklePivot(plantKnee);
    const pushAnkle = _getAnklePivot(pushKnee);

    // Plant leg: firm grip, deep bend
    plantHip.rotation.x = -0.15 * intensity;
    plantKnee.rotation.x = 0.4 * intensity;
    plantAnkle.rotation.x = 0.1 * intensity;

    // Push-off leg: extends
    pushHip.rotation.x = 0.3 * intensity;
    pushKnee.rotation.x = 0.1 * intensity;
    pushAnkle.rotation.x = -0.2 * intensity;

    return true;
}

// ── Idle Animation ──────────────────────────────────────────
function _animateIdle(leftHip, rightHip, now, i) {
    const shift = Math.sin(now * 0.001 + i * 2.0) * 0.05;

    leftHip.rotation.x = 0.02 + shift;
    rightHip.rotation.x = 0.02 - shift;

    const lKnee = _getKneePivot(leftHip);
    const rKnee = _getKneePivot(rightHip);
    lKnee.rotation.x = 0.12 - shift * 0.5;
    rKnee.rotation.x = 0.12 + shift * 0.5;

    _getAnklePivot(lKnee).rotation.x = -0.07;
    _getAnklePivot(rKnee).rotation.x = -0.07;
}

// ── Public API ────────────────────────────────────────────────
function init(container) {
    _createScene(container);
    _createField();
    _createGoals();
    _createAgents();
    _createBall();
    _createTrails();
    _createArrows();
    _animate();
    window.addEventListener('resize', () => _onResize(container));
}

function update(state) {
    targetState = state;
    // Build target map by agent id
    targetAgentMap = {};
    for (const a of state.agents) {
        targetAgentMap[a.id] = {
            x: a.x, y: a.y, vx: a.vx, vy: a.vy, team: a.team,
            facing: a.facing || 0, kicking: a.kicking || 0,
            kick_leg: a.kick_leg || 0, dribbling: a.dribbling || false,
            is_juking: a.is_juking || false,
            kick_power: a.kick_power || 0,
            contact_type: a.contact_type || null,
            z: a.z || 0,
        };
    }
    targetBall = state.ball;

    if (!currentBall) {
        // First state — snap immediately
        currentAgentMap = JSON.parse(JSON.stringify(targetAgentMap));
        currentBall = JSON.parse(JSON.stringify(targetBall));
    }

    // Add newly active agents immediately
    for (const id in targetAgentMap) {
        if (!(id in currentAgentMap)) {
            currentAgentMap[id] = JSON.parse(JSON.stringify(targetAgentMap[id]));
        }
    }
    // Remove agents no longer active
    for (const id in currentAgentMap) {
        if (!(id in targetAgentMap)) {
            delete currentAgentMap[id];
        }
    }
}

function showCelebration(team) {
    _createCelebration(team);
}

// Export as window global for app.js
window.Arena = { init, update, showCelebration };

// ── Scene Setup ───────────────────────────────────────────────
function _createScene(container) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050510);
    scene.fog = new THREE.FogExp2(0x050510, 0.01);

    // Camera — elevated isometric view
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 300);
    camera.position.set(FIELD_W / 2, 24, FIELD_H / 2 + 18);
    camera.lookAt(FIELD_W / 2, 0, FIELD_H / 2);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    // Post-processing: bloom
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(container.clientWidth, container.clientHeight),
        1.2,   // strength
        0.4,   // radius
        0.15   // threshold
    );
    composer.addPass(bloomPass);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(FIELD_W / 2, 0, FIELD_H / 2);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2.2;
    controls.minDistance = 5;
    controls.maxDistance = 50;
    controls.update();

    // Ambient light
    scene.add(new THREE.AmbientLight(0x222233, 0.5));

    // Directional light from above
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight.position.set(FIELD_W / 2, 20, FIELD_H / 2);
    scene.add(dirLight);
}

function _onResize(container) {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    composer.setSize(w, h);
}

// ── Field ─────────────────────────────────────────────────────
function _createField() {
    // Dark ground plane
    const groundGeo = new THREE.PlaneGeometry(FIELD_W + 4, FIELD_H + 4);
    const groundMat = new THREE.MeshStandardMaterial({
        color: 0x080812,
        roughness: 0.9,
        metalness: 0.1,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(FIELD_W / 2, -0.01, FIELD_H / 2);
    scene.add(ground);

    // Field lines (emissive for bloom glow)
    const lineMat = new THREE.LineBasicMaterial({ color: CYAN, transparent: true, opacity: 0.6 });

    // Outline
    _addLine(lineMat, [
        [0, 0], [FIELD_W, 0], [FIELD_W, FIELD_H], [0, FIELD_H], [0, 0]
    ]);

    // Center line
    _addLine(lineMat, [[FIELD_W / 2, 0], [FIELD_W / 2, FIELD_H]]);

    // Center circle
    const circleGeo = new THREE.RingGeometry(3.3, 3.5, 48);
    const circleMat = new THREE.MeshBasicMaterial({ color: CYAN, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
    const circle = new THREE.Mesh(circleGeo, circleMat);
    circle.rotation.x = -Math.PI / 2;
    circle.position.set(FIELD_W / 2, 0.01, FIELD_H / 2);
    scene.add(circle);

    // Center dot
    const dotGeo = new THREE.CircleGeometry(0.2, 16);
    const dotMat = new THREE.MeshBasicMaterial({ color: CYAN, transparent: true, opacity: 0.6 });
    const dot = new THREE.Mesh(dotGeo, dotMat);
    dot.rotation.x = -Math.PI / 2;
    dot.position.set(FIELD_W / 2, 0.01, FIELD_H / 2);
    scene.add(dot);

    // Penalty areas (proportional to field)
    const paW = 4.5, paH = 8.0;
    _addLine(lineMat, [
        [0, (FIELD_H - paH) / 2], [paW, (FIELD_H - paH) / 2],
        [paW, (FIELD_H + paH) / 2], [0, (FIELD_H + paH) / 2]
    ]);
    _addLine(lineMat, [
        [FIELD_W, (FIELD_H - paH) / 2], [FIELD_W - paW, (FIELD_H - paH) / 2],
        [FIELD_W - paW, (FIELD_H + paH) / 2], [FIELD_W, (FIELD_H + paH) / 2]
    ]);

    // Subtle grid on field surface
    const gridMat = new THREE.LineBasicMaterial({ color: CYAN, transparent: true, opacity: 0.04 });
    for (let x = 0; x <= FIELD_W; x += 2) {
        _addLine(gridMat, [[x, 0], [x, FIELD_H]]);
    }
    for (let y = 0; y <= FIELD_H; y += 2) {
        _addLine(gridMat, [[0, y], [FIELD_W, y]]);
    }
}

function _addLine(material, points2D) {
    const pts = points2D.map(p => new THREE.Vector3(p[0], 0.02, p[1]));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.Line(geo, material));
}

// ── Goals ─────────────────────────────────────────────────────
function _createGoals() {
    const goalDepth = 0.8;
    const goalHeight = 1.5;  // proportional to player height (~1.05 to top of body)
    const gy1 = (FIELD_H - GOAL_W) / 2;
    const gy2 = (FIELD_H + GOAL_W) / 2;

    // Left goal (cyan defends — so goal structure is cyan)
    _createGoalStructure(-goalDepth, gy1, goalDepth, GOAL_W, goalHeight, CYAN);
    // Right goal (magenta defends)
    _createGoalStructure(FIELD_W, gy1, goalDepth, GOAL_W, goalHeight, MAGENTA);
}

function _createGoalStructure(x, z, depth, width, height, color) {
    const mat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.8 });

    // Posts and crossbar
    const lines = [
        // Left post
        [[x, 0, z], [x, height, z]],
        // Right post
        [[x, 0, z + width], [x, height, z + width]],
        // Crossbar
        [[x, height, z], [x, height, z + width]],
        // Back posts
        [[x + depth, 0, z], [x + depth, height, z]],
        [[x + depth, 0, z + width], [x + depth, height, z + width]],
        // Back crossbar
        [[x + depth, height, z], [x + depth, height, z + width]],
        // Top depth connectors
        [[x, height, z], [x + depth, height, z]],
        [[x, height, z + width], [x + depth, height, z + width]],
    ];

    lines.forEach(([a, b]) => {
        const geo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(...a), new THREE.Vector3(...b)
        ]);
        scene.add(new THREE.Line(geo, mat));
    });

    // Glow plane on ground inside goal
    const glowGeo = new THREE.PlaneGeometry(depth, width);
    const glowMat = new THREE.MeshBasicMaterial({
        color: color, transparent: true, opacity: 0.08, side: THREE.DoubleSide,
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.rotation.x = -Math.PI / 2;
    glow.position.set(x + depth / 2, 0.02, z + width / 2);
    scene.add(glow);

    // Back net — semi-transparent plane
    const netMat = new THREE.MeshBasicMaterial({
        color: color, transparent: true, opacity: 0.04, side: THREE.DoubleSide,
    });
    const backNetGeo = new THREE.PlaneGeometry(width, height);
    const backNet = new THREE.Mesh(backNetGeo, netMat);
    backNet.position.set(x + depth, height / 2, z + width / 2);
    backNet.rotation.y = Math.PI / 2;
    scene.add(backNet);

    // Side nets
    const sideNetGeo = new THREE.PlaneGeometry(depth, height);
    const sideNet1 = new THREE.Mesh(sideNetGeo, netMat.clone());
    sideNet1.position.set(x + depth / 2, height / 2, z);
    scene.add(sideNet1);
    const sideNet2 = new THREE.Mesh(sideNetGeo, netMat.clone());
    sideNet2.position.set(x + depth / 2, height / 2, z + width);
    scene.add(sideNet2);

    // Top net
    const topNetGeo = new THREE.PlaneGeometry(depth, width);
    const topNet = new THREE.Mesh(topNetGeo, netMat.clone());
    topNet.rotation.x = -Math.PI / 2;
    topNet.position.set(x + depth / 2, height, z + width / 2);
    scene.add(topNet);
}

// ── Agents ────────────────────────────────────────────────────
function _buildLeg(legMat, bootMat, side) {
    // Hip pivot — rotates on X axis at hip joint
    const hipPivot = new THREE.Group();
    hipPivot.position.set(side * LEG_OFFSET_X, HIP_HEIGHT, 0);

    // [0] Thigh cylinder
    const thighGeo = new THREE.CylinderGeometry(
        THIGH_RADIUS_TOP, THIGH_RADIUS_BOT, THIGH_LENGTH, 8
    );
    const thigh = new THREE.Mesh(thighGeo, legMat);
    thigh.position.y = -THIGH_LENGTH / 2;
    hipPivot.add(thigh);

    // [1] Knee pivot — positioned at bottom of thigh
    const kneePivot = new THREE.Group();
    kneePivot.position.y = -THIGH_LENGTH;

    // [1][0] Shin cylinder
    const shinGeo = new THREE.CylinderGeometry(
        SHIN_RADIUS_TOP, SHIN_RADIUS_BOT, SHIN_LENGTH, 8
    );
    const shin = new THREE.Mesh(shinGeo, legMat.clone());
    shin.position.y = -SHIN_LENGTH / 2;
    kneePivot.add(shin);

    // [1][1] Ankle pivot — positioned at bottom of shin
    const anklePivot = new THREE.Group();
    anklePivot.position.y = -SHIN_LENGTH;

    // Boot — box shape with slight forward offset for toe direction
    const bootGeo = new THREE.BoxGeometry(BOOT_WIDTH, BOOT_HEIGHT, BOOT_LENGTH);
    const boot = new THREE.Mesh(bootGeo, bootMat);
    boot.position.set(0, -BOOT_HEIGHT / 2, BOOT_LENGTH * 0.15);
    anklePivot.add(boot);

    kneePivot.add(anklePivot);
    hipPivot.add(kneePivot);

    return hipPivot;
}

function _createAgents() {
    for (let i = 0; i < 4; i++) {
        const team = i < 2 ? 0 : 1;
        const color = team === 0 ? CYAN : MAGENTA;
        const darkColor = color.clone().multiplyScalar(0.6);

        const group = new THREE.Group();

        // child 0: Main body — raised to clear articulated legs
        const bodyGeo = new THREE.SphereGeometry(AGENT_RADIUS, 24, 16);
        const bodyMat = new THREE.MeshStandardMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.6,
            roughness: 0.3,
            metalness: 0.7,
        });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.position.y = AGENT_RADIUS + BODY_RAISE;
        group.add(body);

        // child 1: Inner ring (number indicator)
        const ringGeo = new THREE.TorusGeometry(AGENT_RADIUS * 0.35, 0.03, 8, 24);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.y = AGENT_RADIUS + BODY_RAISE;
        ring.rotation.x = Math.PI / 2;
        group.add(ring);

        // child 2: Point light for local glow
        const light = new THREE.PointLight(color, 0.5, 3);
        light.position.y = AGENT_RADIUS + BODY_RAISE;
        group.add(light);

        // child 3: Shadow disc on ground
        const shadowGeo = new THREE.CircleGeometry(AGENT_RADIUS * 0.8, 16);
        const shadowMat = new THREE.MeshBasicMaterial({
            color: color, transparent: true, opacity: 0.15,
        });
        const shadow = new THREE.Mesh(shadowGeo, shadowMat);
        shadow.rotation.x = -Math.PI / 2;
        shadow.position.y = 0.01;
        group.add(shadow);

        // child 4: Legs container (rotates Y to face movement direction)
        const legsContainer = new THREE.Group();

        const legMat = new THREE.MeshStandardMaterial({
            color: darkColor,
            emissive: color,
            emissiveIntensity: 0.3,
            roughness: 0.4,
            metalness: 0.6,
        });

        const bootMat = new THREE.MeshStandardMaterial({
            color: 0x111111,
            emissive: color,
            emissiveIntensity: 0.15,
            roughness: 0.6,
            metalness: 0.3,
        });

        // Left leg (child 0 of legsContainer)
        legsContainer.add(_buildLeg(legMat, bootMat, -1));
        // Right leg (child 1 of legsContainer)
        legsContainer.add(_buildLeg(legMat.clone(), bootMat.clone(), 1));

        group.add(legsContainer);

        scene.add(group);
        agentMeshes.push(group);
    }
}

// ── Ball ──────────────────────────────────────────────────────
function _createBall() {
    const group = new THREE.Group();

    const ballGeo = new THREE.SphereGeometry(BALL_RADIUS, 20, 12);
    const ballMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: 0xffffff,
        emissiveIntensity: 0.8,
        roughness: 0.2,
        metalness: 0.5,
    });
    const ball = new THREE.Mesh(ballGeo, ballMat);
    ball.position.y = BALL_RADIUS;
    group.add(ball);

    // Ball glow light
    const light = new THREE.PointLight(0xffffcc, 0.6, 4);
    light.position.y = BALL_RADIUS;
    group.add(light);

    // Shadow disc
    const shadowGeo = new THREE.CircleGeometry(BALL_RADIUS * 0.7, 12);
    const shadowMat = new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.1,
    });
    const shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = 0.01;
    group.add(shadow);

    scene.add(group);
    ballMesh = group;
}

// ── Trails ────────────────────────────────────────────────────
function _createTrails() {
    for (let i = 0; i < 4; i++) {
        const team = i < 2 ? 0 : 1;
        const color = team === 0 ? CYAN : MAGENTA;
        trailMeshes.agents[i] = [];
        for (let t = 0; t < MAX_TRAIL; t++) {
            const geo = new THREE.SphereGeometry(AGENT_RADIUS * 0.2, 8, 4);
            const mat = new THREE.MeshBasicMaterial({
                color: color, transparent: true,
                opacity: (t + 1) / MAX_TRAIL * 0.25,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.visible = false;
            mesh.position.y = 0.1;
            scene.add(mesh);
            trailMeshes.agents[i].push(mesh);
        }
    }
    // Ball trail
    for (let t = 0; t < MAX_TRAIL; t++) {
        const geo = new THREE.SphereGeometry(BALL_RADIUS * 0.3, 8, 4);
        const mat = new THREE.MeshBasicMaterial({
            color: 0xffffff, transparent: true,
            opacity: (t + 1) / MAX_TRAIL * 0.3,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.visible = false;
        mesh.position.y = 0.1;
        scene.add(mesh);
        trailMeshes.ball.push(mesh);
    }
}

// Trail position history
const trailHistory = {
    agents: [[], [], [], []],
    ball: [],
};

function _updateTrails() {
    if (!currentBall) return;

    // Update agent trail history — indexed by agent id (0-3)
    for (let i = 0; i < 4; i++) {
        const a = currentAgentMap[i];
        if (a) {
            trailHistory.agents[i].push({ x: a.x, y: a.y });
            if (trailHistory.agents[i].length > MAX_TRAIL) trailHistory.agents[i].shift();

            trailHistory.agents[i].forEach((p, t) => {
                const mesh = trailMeshes.agents[i][t];
                if (mesh) {
                    mesh.position.x = p.x;
                    mesh.position.z = p.y;
                    mesh.visible = true;
                }
            });
        } else {
            // Hide trails for inactive agents
            trailHistory.agents[i] = [];
            trailMeshes.agents[i].forEach(mesh => { mesh.visible = false; });
        }
    }

    // Ball trail
    trailHistory.ball.push({ x: currentBall.x, y: currentBall.y });
    if (trailHistory.ball.length > MAX_TRAIL) trailHistory.ball.shift();
    trailHistory.ball.forEach((p, t) => {
        const mesh = trailMeshes.ball[t];
        if (mesh) {
            mesh.position.x = p.x;
            mesh.position.z = p.y;
            mesh.visible = true;
        }
    });
}

// ── Arrows ────────────────────────────────────────────────────
function _createArrows() {
    for (let i = 0; i < 4; i++) {
        const team = i < 2 ? 0 : 1;
        const color = team === 0 ? CYAN : MAGENTA;
        const arrow = new THREE.ArrowHelper(
            new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0.3, 0),
            1, color.getHex(), 0.15, 0.1
        );
        arrow.visible = false;
        scene.add(arrow);
        arrowHelpers.push(arrow);
    }
}

function _updateArrows() {
    if (!currentBall) return;
    for (let i = 0; i < 4; i++) {
        const arrow = arrowHelpers[i];
        if (!arrow) continue;
        const a = currentAgentMap[i];
        if (!a) {
            arrow.visible = false;
            continue;
        }
        const speed = Math.sqrt(a.vx * a.vx + a.vy * a.vy);
        if (speed > 0.5) {
            arrow.visible = true;
            arrow.position.set(a.x, 0.3, a.y);
            const dir = new THREE.Vector3(a.vx / speed, 0, a.vy / speed);
            arrow.setDirection(dir);
            arrow.setLength(Math.min(speed * 0.3, 2), 0.15, 0.1);
        } else {
            arrow.visible = false;
        }
    }
}

// ── Celebration Particles ─────────────────────────────────────
function _createCelebration(team) {
    // Remove old
    if (celebrationParticles) {
        scene.remove(celebrationParticles);
        celebrationParticles.geometry.dispose();
        celebrationParticles.material.dispose();
    }

    const color = team === 'cyan' ? CYAN : MAGENTA;
    const count = 200;
    const positions = new Float32Array(count * 3);
    const velocities = [];

    // Goal position
    const gx = team === 'cyan' ? FIELD_W : 0;
    const gz = FIELD_H / 2;

    for (let i = 0; i < count; i++) {
        positions[i * 3] = gx;
        positions[i * 3 + 1] = 0.5;
        positions[i * 3 + 2] = gz;
        velocities.push({
            x: (Math.random() - 0.5) * 8,
            y: Math.random() * 6 + 2,
            z: (Math.random() - 0.5) * 8,
        });
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
        color: color, size: 0.15, transparent: true, opacity: 1.0,
        blending: THREE.AdditiveBlending,
    });
    celebrationParticles = new THREE.Points(geo, mat);
    celebrationParticles._velocities = velocities;
    celebrationParticles._life = 2.0;
    celebrationClock = 0;
    scene.add(celebrationParticles);
}

function _updateCelebration(dt) {
    if (!celebrationParticles) return;

    celebrationClock += dt;
    if (celebrationClock > celebrationParticles._life) {
        scene.remove(celebrationParticles);
        celebrationParticles.geometry.dispose();
        celebrationParticles.material.dispose();
        celebrationParticles = null;
        return;
    }

    const positions = celebrationParticles.geometry.attributes.position.array;
    const vels = celebrationParticles._velocities;

    for (let i = 0; i < vels.length; i++) {
        positions[i * 3] += vels[i].x * dt;
        positions[i * 3 + 1] += vels[i].y * dt;
        positions[i * 3 + 2] += vels[i].z * dt;
        vels[i].y -= 9.8 * dt; // gravity
    }
    celebrationParticles.geometry.attributes.position.needsUpdate = true;
    celebrationParticles.material.opacity = 1.0 - (celebrationClock / celebrationParticles._life);
}

// ── Animation Loop ────────────────────────────────────────────
const clock = new THREE.Clock();

function _animate() {
    requestAnimationFrame(_animate);
    const dt = Math.min(clock.getDelta(), 0.1);

    // Interpolate current state toward target
    if (targetBall && currentBall) {
        for (const id in currentAgentMap) {
            const a = currentAgentMap[id];
            const t = targetAgentMap[id];
            if (t) {
                a.x += (t.x - a.x) * LERP_SPEED;
                a.y += (t.y - a.y) * LERP_SPEED;
                a.vx = t.vx;
                a.vy = t.vy;
                // Copy non-interpolatable state
                a.kicking = t.kicking;
                a.kick_leg = t.kick_leg;
                a.dribbling = t.dribbling;
                a.facing = t.facing;
                a.is_juking = t.is_juking;
                a.kick_power = t.kick_power;
                a.contact_type = t.contact_type;
                a.z = (a.z || 0) + ((t.z || 0) - (a.z || 0)) * LERP_SPEED;
            }
        }
        currentBall.x += (targetBall.x - currentBall.x) * LERP_SPEED;
        currentBall.y += (targetBall.y - currentBall.y) * LERP_SPEED;
        currentBall.vx = targetBall.vx;
        currentBall.vy = targetBall.vy;
        // 3D ball height — interpolate smoothly
        const tz = targetBall.z || 0;
        currentBall.z = (currentBall.z || 0) + (tz - (currentBall.z || 0)) * LERP_SPEED;
        currentBall.vz = targetBall.vz || 0;
    }

    const now = performance.now();

    // Update 3D objects — show/hide based on active agents, animate legs
    for (let i = 0; i < 4; i++) {
        const group = agentMeshes[i];
        if (!group) continue;
        const a = currentAgentMap[i];
        if (a) {
            group.visible = true;
            group.position.x = a.x;
            group.position.z = a.y;
            const agentZ = a.z || 0;
            group.position.y = agentZ;  // vertical offset from jumping

            // Shadow stays on ground — counter-offset and scale with height
            const shadow = group.children[3];
            shadow.position.y = 0.01 - agentZ;  // cancel out group Y offset
            const shScale = Math.max(0.3, 1.0 - agentZ * 0.4);
            shadow.scale.set(shScale, shScale, 1);
            shadow.material.opacity = Math.max(0.04, 0.15 - agentZ * 0.06);

            const bob = Math.sin(now * 0.003 + i * 1.5) * 0.03;
            group.children[0].position.y = AGENT_RADIUS + BODY_RAISE + bob;

            const speed = Math.sqrt(a.vx * a.vx + a.vy * a.vy);
            const body = group.children[0];
            const anim = agentAnimState[i];

            // ── Dribble glow: pulse emissive when dribbling ──
            if (body.material) {
                body.material.emissiveIntensity = a.dribbling
                    ? 0.8 + Math.sin(now * 0.01) * 0.2
                    : 0.6;
            }

            // ── Articulated leg animation ──
            const legsContainer = group.children[4];
            if (legsContainer) {
                // Face movement direction (smooth rotation)
                if (speed > 0.3) {
                    const targetAngle = Math.atan2(a.vx, a.vy);
                    let angleDiff = targetAngle - legsContainer.rotation.y;
                    while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
                    while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;
                    legsContainer.rotation.y += angleDiff * 0.15;
                }

                const leftHip = legsContainer.children[0];
                const rightHip = legsContainer.children[1];
                const kicking = a.kicking || 0;

                // Update juke timer from backend event
                if (a.is_juking) {
                    anim.jukeTimer = 1.0;
                    anim.jukeSide = (a.vx > 0) ? 1 : -1;
                }
                anim.jukeTimer *= 0.92;
                if (anim.jukeTimer < 0.05) anim.jukeTimer = 0;

                // Priority: kick > jump tuck > juke > dribble > walk > idle
                const agentZ = a.z || 0;
                if (kicking > 0.1) {
                    const ct = a.contact_type;
                    if (ct === 'header') {
                        _animateHeader(body, kicking);
                        _animateIdle(leftHip, rightHip, now, i);
                    } else if (ct === 'knee') {
                        _animateKneeStrike(leftHip, rightHip, a.kick_leg, kicking, a.kick_power || 0.5);
                    } else {
                        _animateKick(leftHip, rightHip, a.kick_leg, kicking, a.kick_power || 0.5);
                    }
                } else if (agentZ > 0.05) {
                    _animateJumpTuck(leftHip, rightHip, agentZ);
                } else if (_animateJuke(leftHip, rightHip, anim.jukeTimer, anim.jukeSide)) {
                    // juke handled — body lean into direction
                    body.position.z = -0.04 * anim.jukeSide * anim.jukeTimer;
                } else if (a.dribbling && speed > 0.5) {
                    _animateDribble(leftHip, rightHip, speed, now);
                    body.position.z = -0.05;  // forward lean
                } else if (speed > 0.3) {
                    _animateWalk(leftHip, rightHip, speed, now);
                    body.position.z = 0;
                } else {
                    _animateIdle(leftHip, rightHip, now, i);
                    body.position.z = 0;
                }
            }
        } else {
            group.visible = false;
        }
    }

    if (ballMesh && currentBall) {
        ballMesh.position.x = currentBall.x;
        ballMesh.position.z = currentBall.y;

        // 3D ball height — ball sphere + glow rise, shadow stays on ground
        const ballZ = currentBall.z || 0;
        ballMesh.children[0].position.y = BALL_RADIUS + ballZ;  // ball sphere
        ballMesh.children[1].position.y = BALL_RADIUS + ballZ;  // glow light
        // Shadow (children[2]) stays at y=0.01, scales down with height
        const shadowScale = Math.max(0.3, 1.0 - ballZ * 0.3);
        ballMesh.children[2].scale.set(shadowScale, shadowScale, 1);
        ballMesh.children[2].material.opacity = Math.max(0.03, 0.1 - ballZ * 0.03);

        const speed = Math.sqrt(currentBall.vx ** 2 + currentBall.vy ** 2);
        const vz = currentBall.vz || 0;
        const totalSpeed = Math.sqrt(speed * speed + vz * vz);
        if (totalSpeed > 0.1) {
            ballMesh.children[0].rotation.x += totalSpeed * dt * 2;
            ballMesh.children[0].rotation.z += speed * dt;
        }
    }

    _updateTrails();
    _updateArrows();
    _updateCelebration(dt);

    controls.update();
    composer.render();
}
