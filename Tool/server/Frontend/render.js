/** render the canvas and timeline */
const BACKGROUND_COLOR = 0xffffff;
// Constants relating to the camera parameters.
const PERSP_CAMERA_FOV_VERTICAL = 70;
const PERSP_CAMERA_NEAR_CLIP_PLANE = 0.01;
const PERSP_CAMERA_FAR_CLIP_PLANE = 100;
const ORTHO_CAMERA_FRUSTUM_HALF_EXTENT = 1.2;
const MIN_ZOOM_SCALE = 0.2;
const MAX_ZOOM_SCALE = 60;
const NORMAL_SIZE = 10;
const HOVER_SIZE = 22;
const SELECTED_SIZE = 15;
const GRAY = [0.8, 0.8, 0.8];
const DEFAULT_ALPHA = 1.0;  // less than 1.0 will cause raycaster no intersection
const SELECTED_ALPHA = 1.0;
const selectedLabel = 'fixedHoverLabel';
let baseZoomSpeed = 0.01;
let isDragging = false;
let previousMousePosition = {
    x: 0,
    y: 0
};
let lockIndex = false;

const p3d = (idx, posArr) => {
    let x = posArr.getX(idx);
    let y = posArr.getY(idx);
    let z = posArr.getZ(idx);
    return { x, y, z };
};

const lineTwoPoints = (ctx, p1, p2) => {
    let points = [];
    points.push(new THREE.Vector3(p1.x, p1.y, p1.z));
    points.push(new THREE.Vector3(p2.x, p2.y, p2.z));
    let geometry = new THREE.BufferGeometry().setFromPoints(points);
    ctx.scene.hoverLine = new THREE.Line(geometry, ctx.lineMaterial);
    ctx.lines.push(ctx.scene.hoverLine);
    ctx.scene.add(ctx.scene.hoverLine);
};

const removeAllLines = (ctx, lines) => {
    lines.forEach(line => {
        ctx.scene.remove(line);
        line.geometry.dispose();
    });
    lines.length = 0;
}

const setSizeAndAlpha = (index, sizeArr, alphaArr, isEmphasize = false) => {
    sizeArr[index] = HOVER_SIZE;
    if (isEmphasize) {
        alphaArr[index] = SELECTED_ALPHA;
    }
}

const revealNeighborPoints = (ctx, index, relArr, sizeArr, alphaArr, posArr, isEmphasize = false) => {
    relArr[index].forEach(neighbor => {
        setSizeAndAlpha(neighbor, sizeArr, alphaArr, isEmphasize);
        lineTwoPoints(ctx, p3d(index, posArr), p3d(neighbor, posArr));
    });
};

const revealPoint = (ctx, index, relArr, sizeArr, alphaArr, posArr, isEmphasize = false) => {
    if (index !== null && index !== undefined) {
        setSizeAndAlpha(index, sizeArr, alphaArr, isEmphasize);
        if (relArr) {
            revealNeighborPoints(ctx, index, relArr, sizeArr, alphaArr, posArr, isEmphasize);
        }
    }
}

const updateHoveredIndexSize = (ctx, hoveredIndex, selectedIndices, highlightAttributes, nnIndices) => {
    const sizeArr = ctx.pointsMesh.geometry.attributes.size.array;
    const posArr = ctx.pointsMesh.geometry.attributes.position;    // keep dimension
    const alphaArr = ctx.pointsMesh.geometry.attributes.alpha.array;

    // TODO Can incremental update here optimize the performance?
    for (let i = 0; i < sizeArr.length; i++) {
        sizeArr[i] = NORMAL_SIZE;
    }
    removeAllLines(ctx, ctx.lines);

    // TODO This logic should not be put here. Just for temporary test.
    // What points are revealed should be determined in 'model' but not 'vision'.
    // Especially like locked index should be preserved somewhere else
    let top_k = undefined;
    if (ctx.vueApp.taskType === 'Umap-Neighborhood') {
        const top_k_attr_name =
            ctx.vueApp.neighborhoodRevealType === 'Intra-Type' ? 'intra_sim_top_k'
            : ctx.vueApp.neighborhoodRevealType === 'Inter-Type' ? 'inter_sim_top_k'
            : undefined;
        if (top_k_attr_name) {
            top_k = ctx.vueApp.epochData[top_k_attr_name];
        } else if (this.vueApp.neighborhoodRevealType === 'Both') {
            const zip = (arr1, arr2) => {
                return arr1.map((k, i) => [...k, ...arr2[i]]);
            }
            top_k = zip(ctx.vueApp.epochData['intra_sim_top_k'], ctx.vueApp.epochData['inter_sim_top_k']);
        }
    }

    revealPoint(ctx, hoveredIndex, top_k, sizeArr, alphaArr, posArr, false);
    selectedIndices.forEach((index) => revealPoint(ctx, index, top_k, sizeArr, alphaArr, posArr, true));

    ctx.pointsMesh.geometry.attributes.size.needsUpdate = true;        // TODO Is this a drawback of performance? We mark the whole array as needing update
    ctx.pointsMesh.geometry.attributes.position.needsUpdate = true;
    ctx.pointsMesh.geometry.attributes.alpha.needsUpdate = true;
}

class PlotCanvas {
    constructor(vueApp) {
        // bind attributes to a vue app
        this.vueApp = vueApp;
        this.eventListeners = [];
        this.animations = [];
    }

    getScreenPositionOfPoint(index) {
        const vector = new THREE.Vector3().fromBufferAttribute(this.pointsMesh.geometry.attributes.position, index);
        return worldPositionToScreenPosition(vector, this.camera, this.canvas.getBoundingClientRect());
    }

    // bind to a container, initiating a scene in it
    bindTo(container) {
        container.innerHTML = "";
        this.container = container;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        const rect = container.getBoundingClientRect();
        renderer.setSize(rect.width, rect.height);
        renderer.setClearColor(BACKGROUND_COLOR, 1);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.toneMapping = false;
        this.renderer = renderer;

        this.canvas = renderer.domElement;
        container.appendChild(renderer.domElement);

        // const logAndValidate = (functionName, args) => {
        //     logGLCall(functionName, args);
        //     validateNoneOfTheArgsAreUndefined (functionName, args);
        // }

        // const throwOnGLError = (err, funcName, args) => {
        //     throw WebGLDebugUtils.glEnumToString(err) + " was caused by call to: " + funcName;
        // };

        // let gl = this.canvas.getContext('webgl');
        // gl = WebGLDebugUtils.makeDebugContext(gl, throwOnGLError, logAndValidate);
    }

    plotDataPoints(visData, updateOnly = false) {
        const x_center = (visData.grid_index[0] + visData.grid_index[2]) / 2;
        const y_center = (visData.grid_index[1] + visData.grid_index[3]) / 2;
        const x_scale = 12 / (visData.grid_index[2] - visData.grid_index[0]);
        const y_scale = 12 / (visData.grid_index[3] - visData.grid_index[1]);

        visData.result = visData.result.map(([x, y]) => [(x - x_center) * x_scale, (y - y_center) * y_scale]);

        const boundary = {
            x_min: (visData.grid_index[0] - x_center) * x_scale,
            y_min: (visData.grid_index[1] - y_center) * y_scale,
            x_max: (visData.grid_index[2] - x_center) * x_scale,
            y_max: (visData.grid_index[3] - y_center) * y_scale
        };

        this.__updatePlotBoundary(boundary);
        if (updateOnly) {
            this.animations = [];
            this.__updateDataPointsPosition(visData);
        } else {
            this.__initScene();
            this.__initCamera();
            // this.__putPlane(visData.grid_color);
            this.__putDataPoints(visData);
            this.__addControls();
        }
        this.__syncAttributesToVueApp();
    }

    __syncAttributesToVueApp() {
        this.vueApp.renderer = this.renderer;
        this.vueApp.scene = this.scene;
        this.vueApp.camera = this.camera;
        this.vueApp.pointsMesh = this.pointsMesh;
    }

    __getObjectPositionInWindow(pointMesh, i) {
        const vector = new THREE.Vector3().fromBufferAttribute(pointMesh.geometry.attributes.position, i);

        vector.project(this.camera); // `camera` is a THREE.PerspectiveCamera

        // https://stackoverflow.com/a/73028232/17760236, should use browser-computed width and height
        const { width, height } = this.canvas.getBoundingClientRect();
        const x = Math.round((0.5 + vector.x / 2) * width);
        const y = Math.round((0.5 - vector.y / 2) * height);
        return [x, y];
    }

    __syncPointMeshPositions() {
        const positionAttribute = this.pointsMesh.geometry.getAttribute('position');
        const size = positionAttribute.count;

        const result = [];
        for (let i = 0; i < size; ++i) {
            const coord = this.__getObjectPositionInWindow(this.pointsMesh, i);
            result.push(coord);
        }

        this.vueApp.pointCoords = result;
    }

    render() {
        const animate = () => {
            const frameId = requestAnimationFrame(animate);
            this.vueApp.animationFrameId = frameId
            this.animations = this.animations.filter((a) => a(frameId));
            this.renderer.render(this.scene, this.camera);
            this.__syncPointMeshPositions();
        }
        animate();
    }

    disposeResources() {
        this.eventListeners.forEach((e) => {
            const [type, listener] = e;
            this.container?.removeEventListener(type, listener);
        })
        this.eventListeners.length = 0;

        clearTimeout(this.renderTimeout);
        if (this.vueApp.animationFrameId) {
            cancelAnimationFrame(this.vueApp.animationFrameId);
            this.vueApp.animationFrameId = undefined;
        }

        if (this.scene) {
            this.scene.traverse((object) => {
                if (!object.isMesh) return;

                object.geometry.dispose();

                if (object.material.isMaterial) {
                    cleanMaterial(object.material);
                } else {
                    // an array of materials
                    for (const material of object.material) cleanMaterial(material);
                }
            });

            function cleanMaterial(material) {
                material.dispose();

                // dispose textures
                for (const key in material) {
                    if (material[key] && material[key].isTexture) {
                        material[key].dispose();
                    }
                }
            }

            this.scene.dispose?.();
        }

        this.renderer?.dispose?.();
    }

    __updatePlotBoundary(boundary) {
        this.boundary = boundary;
        // Object.assign(this.sceneBoundary, boundary);
    }

    __initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(BACKGROUND_COLOR);

        // render scene as all-white
        let ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
        this.scene.add(ambientLight);
    }

    __initCamera() {
        const rect = this.container.getBoundingClientRect();

        // A non-one aspectRatio will result in not-fitting error:
        // let aspectRatio = rect.width / rect.height;
        let aspectRatio = 1;

        const camera = new THREE.OrthographicCamera(
            this.boundary.x_min * aspectRatio,
            this.boundary.x_max * aspectRatio,
            this.boundary.y_max,
            this.boundary.y_min,
            1, 1000);

        const init_x = (this.boundary.x_max + this.boundary.x_min) / 2;
        const init_y = (this.boundary.y_max + this.boundary.y_min) / 2;
        camera.position.set(init_x, init_y, 100);
        camera.lookAt(new THREE.Vector3(0, 0, 0));

        this.camera = camera;
    }

    __putPlane(color) {
        let width = this.boundary.x_max - this.boundary.x_min;
        let height = this.boundary.y_max - this.boundary.y_min;
        let centerX = this.boundary.x_min + width / 2;
        let centerY = this.boundary.y_min + height / 2;

        this.__createPureColorTexture(color, (texture) => {
            let material = new THREE.MeshPhongMaterial({
                map: texture,
                side: THREE.DoubleSide
            });
            let plane_geometry = new THREE.PlaneGeometry(width, height);
            let newMesh = new THREE.Mesh(plane_geometry, material);
            newMesh.position.set(centerX, centerY, 0);
            this.scene.add(newMesh);
        })
    }

    __putDataPoints(visData) {
        // FIXME label_color_list is used in rendering, so color_list does not work anymore. Should accord to which?
        const geoData = visData.result;
        this.geoData = geoData;

        let color = visData.label_list.map((x) => visData.color_list[x]);
        let position = [];
        let colors = [];
        let sizes = [];
        let alphas = [];

        geoData.forEach(function (point, i) {
            position.push(point[0], point[1], 0);
            colors.push(color[i][0] / 255, color[i][1] / 255, color[i][2] / 255);
            sizes.push(NORMAL_SIZE);
            alphas.push(DEFAULT_ALPHA);
        });
        // console.log("datapoints", geoData.length);

        let geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
        geometry.setAttribute('alpha', new THREE.Float32BufferAttribute(alphas, 1));

        let shaderMaterial = this.__getDefaultPointShaderMaterial();

        this.pointsMesh = new THREE.Points(geometry, shaderMaterial);
        this.__savePointMeshSettings();
        this.__updateSelectedPoint();

        this.scene.add(this.pointsMesh);
    }

    __updateDataPointsPosition(visData) {
        const geoData = visData.result;
        let tar = [];

        geoData.forEach(function (point) {
            tar.push(point[0], point[1], 0);
        });

        // returns true if their is move, else false
        const moveNonLinear = (i, cur, tar) => {
            const d = tar[i] - cur[i];
            const dThres = 0.01;
            const grade = 0.2;

            if (d == 0) return;
            let r;
            // add a final linear
            if (d >= dThres || d <= -dThres) {
                r = cur[i] + d * grade;
            } else if (d > 0 && d < dThres) {
                r = cur[i] + dThres * grade;
            } else {
                r = cur[i] - dThres * grade;
            }

            // add quick stop
            if ((r > tar[i]) == (cur[i] > tar[i])) {
                cur[i] = r;
                return true;
            } else {
                cur[i] = tar[i];
                return false;
            }
        }

        // FIXME prevent multiple same-kind animation, but not only one animation
        if (this.animations.length) {
            this.animations.length = 0;
        }

        // FIXME is it soft-rendering here? Could it be slow? Need shader?
        const moveAnimation = () => {
            const ori = this.pointsMesh.geometry.attributes.position.array;
            let moving = false;
            for (let i = 0; i < ori.length; ++i) {
                if (moveNonLinear(i, ori, tar)) moving = true;
            }
            this.lastDoUpdateRevealing?.();
            this.pointsMesh.geometry.attributes.position.needsUpdate = true;
            return (moving);     // only when not done the animation needs to continue
        }
        this.animations.push(moveAnimation);
    }

    __addControls() {
        this.__addClassicMapNavigationControls();
        this.__addHoverRevealingControl();
        this.__addDoubleClickLockingControl();

        this.__addFilterTestTrain();
        this.__addHighlight();
    }

    __createPureColorTexture(color, callback) {
        let canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        let ctx = canvas.getContext("2d");
        ctx.fillStyle = `rgb(255, 255, 255)`;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        let texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        texture.encoding = THREE.sRGBEncoding;
        callback(texture);
    }

    __registerContainerEventListener(type, listener) {
        const targetListener = listener.bind(this);
        this.container.addEventListener(type, targetListener);
        this.eventListeners.push([type, targetListener]);
    }

    __addClassicMapNavigationControls() {
        this.__addWheelZoomingControl();
        this.__addMouseDraggingControl();
    }

    __addWheelZoomingControl() {
        const changeZoom = (event, camera) => {
            const currentZoom = camera.zoom;
            // Assume newZoom = a * b^(times + x0), then a * b^x0 = MIN_ZOOM_SCALE
            const a = 1.0;
            const b = 1.1;
            const x = Math.log(currentZoom / a) / Math.log(b);
            const g = 100;
            const new_x = x - event.deltaY / g;  // wheels down, deltaY is positive, but scaling going down
            let newZoom = a * Math.pow(b, new_x);
            newZoom = Math.max(MIN_ZOOM_SCALE, Math.min(newZoom, MAX_ZOOM_SCALE));
            camera.zoom = newZoom;
            this.__moveCameraWithinRange(camera);
            camera.updateProjectionMatrix();
        }

        this.__registerContainerEventListener('wheel', (event) => {
            changeZoom(event, this.camera);
            event.preventDefault();
        });
    }

    __addMouseDraggingControl() {
        let isDragging = false;

        this.__registerContainerEventListener('mousedown', function (e) {
            if (this.vueApp.SelectionMode && this.vueApp.isShifting) {

            } else {
                isDragging = true;
                if (this.container.style.cursor != 'pointer') {
                    this.container.style.cursor = 'move';
                }
                previousMousePosition.x = e.clientX;
                previousMousePosition.y = e.clientY;
            }
        });

        // handel mouse move
        this.__registerContainerEventListener('mousemove', function (e) {
            if (isDragging) {
                const currentZoom = this.camera.zoom;

                const mouseDeltaX = e.clientX - previousMousePosition.x;
                const mouseDeltaY = e.clientY - previousMousePosition.y;

                const viewportWidth = this.renderer.domElement.clientWidth;
                const viewportHeight = this.renderer.domElement.clientHeight;

                // Scale factors
                const scaleX = (this.camera.right - this.camera.left) / viewportWidth;
                const scaleY = (this.camera.top - this.camera.bottom) / viewportHeight;

                // Convert pixel movement to world units
                const deltaX = (mouseDeltaX * scaleX) / currentZoom;
                const deltaY = (mouseDeltaY * scaleY) / currentZoom;

                // Update the camera position based on the scaled delta
                let newPosX = this.camera.position.x - deltaX * 1;
                let newPosY = this.camera.position.y + deltaY * 1;

                this.__moveCameraWithinRange(this.camera, newPosX, newPosY);

                // update previous mouse position
                previousMousePosition = {
                    x: e.clientX,
                    y: e.clientY
                };
                // updateLabelPosition('', this.vueApp.selectedPointPosition, this.vueApp.selectedIndex, selectedLabel, true)
                updateCurrHoverIndex(e, null, true, '');
            }
        });

        // mouse up event
        this.__registerContainerEventListener('mouseup', function (e) {
            isDragging = false;
            this.container.style.cursor = 'default';
        });
    }

    __getDefaultPointShaderMaterial() {
        if (this.shaderMaterial === undefined) {
            this.shaderMaterial = this.__createPointShaderMaterial();
        }
        return this.shaderMaterial;
    }

    __createPointShaderMaterial() {
        const vertexShader = `
            attribute vec3 color;
            attribute float size;
            varying vec3 vColor;
            void main() {
                vColor = color;  // Pass color to fragment shader
                gl_PointSize = size;  // Use custom size attribute
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            varying vec3 vColor;
            void main() {
                // Calculate distance from center of point
                vec2 uv = gl_PointCoord.xy - 0.5;
                float dist = length(uv);

                // If distance is greater than 0.5 (outside circle), discard fragment
                if (dist > 0.5) discard;

                gl_FragColor = vec4(vColor, 1.0);  // Use varying color
            }
        `;

        // Shader Material
        const material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
        });

        return material;
    }

    __savePointMeshSettings() {
        if (this.pointsMesh.geometry.getAttribute('size')) {
            this.vueApp.originalSettings.originalSizes = Array.from(this.pointsMesh.geometry.getAttribute('size').array);
        }

        if (this.pointsMesh.geometry.getAttribute('color')) {
            this.vueApp.originalSettings.originalColors = Array.from(this.pointsMesh.geometry.getAttribute('color').array);
        }
    }

    // TODO this seems to be useless, functionality replaced by the next method
    __updateSelectedPoint() {
        if (this.vueApp.selectedIndex) {
            this.pointsMesh.geometry.attributes.size.array[this.vueApp.selectedIndex] = HOVER_SIZE;
            let pointPosition = new THREE.Vector3();
            pointPosition.fromBufferAttribute(this.pointsMesh.geometry.attributes.position, this.vueApp.selectedIndex);
            this.vueApp.selectedPointPosition = pointPosition;
            this.pointsMesh.geometry.attributes.size.needsUpdate = true;
        }
    }

    __updateLastHoveredIndex() {
        let specifiedLastHoveredIndex = makeSpecifiedVariableName('curIndex', '');
        let specifiedImageSrc = makeSpecifiedVariableName('imageSrc', '');
        let specifiedSelectedIndex = makeSpecifiedVariableName('selectedIndex', '');
        let specifiedHighlightAttributes = makeSpecifiedVariableName('highlightAttributes', '');

        updateHoveredIndexSize(this, this.vueApp[specifiedLastHoveredIndex], this.vueApp[specifiedSelectedIndex],
            this.vueApp[specifiedHighlightAttributes].visualizationError, this.vueApp.nnIndices, this.pointsMesh, this.vueApp);
    }

    __addHoverRevealingControl() {
        let raycaster = new THREE.Raycaster();
        let mouse = new THREE.Vector2();

        this.lines = [];
        this.lineMaterial = new THREE.LineBasicMaterial({ color: 0xaaaaaa });

        // FIXME it's not a good practice to use function here, because always needs remapping
        function onMouseMove(event, isDown = false) {
            // TODO consider adjusting the threshold reactive to monitor size and resolution
            raycaster.params.Points.threshold = 0.1 / this.camera.zoom; // 根据点的屏幕大小调整
            let rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, this.camera);
            let intersects = raycaster.intersectObject(this.pointsMesh);

            let index = null;
            if (intersects.length > 0 && checkVisibility(this.pointsMesh.geometry.attributes.alpha.array, intersects[0].index)) {

                this.container.style.cursor = 'pointer';

                let intersect = intersects[0];

                index = intersect.index;

                // Map it to the original index
                if (this.vueApp.filter_index != '') {
                    console.log("this.vueApp.filter_index", this.vueApp.filter_index);
                    const filter_index = this.vueApp.filter_index.split(',');
                    index = filter_index[index];
                }

                // This index is deem as hovered

            }
            // If isDown, switch lockIndex
            // lockIndex indicates whether the last one or several indices should be deem as hovered
            if (isDown) {
                // detect if locked to a new point successfully
                if (index !== null && index !== undefined) {
                    if (!this.vueApp[specifiedSelectedIndex].includes(index)) {
                        lockIndex = true;
                        this.vueApp[specifiedSelectedIndex].push(index);
                    } else {
                        this.vueApp[specifiedSelectedIndex] = this.vueApp[specifiedSelectedIndex].filter((value) => value !== index);
                    }
                }
            }

            this.__updateLastHoveredIndex();

            this.lastDoUpdateRevealing = this.__updateLastHoveredIndex.bind(this);
            updateCurrHoverIndex(event, index, false, '', this.getScreenPositionOfPoint(index));

            if ((index === null || index === undefined) && this.container.style.cursor !== 'move') {
                this.container.style.cursor = 'default';
            }


                // if (this.vueApp.lastHoveredIndex !== null && !lockIndex) {
                //     updateHoveredIndexSize(this.vueApp[specifiedLastHoveredIndex], this.vueApp[specifiedSelectedIndex],
                //         this.vueApp[specifiedHighlightAttributes].visualizationError, this.vueApp.nnIndices);
                //     this.pointsMesh.geometry.attributes.size.needsUpdate = true;
                //     this.vueApp.nnIndices = [];
                //     this.vueApp[specifiedLastHoveredIndex] = null;
                //     this.vueApp[specifiedImageSrc] = "";
                //     updateCurrHoverIndex(event, null, false, '');
                //     // if (this.scene.hoverLine) {
                //     //     this.scene.remove(this.scene.hoverLine);
                //     //     this.scene.hoverLine.geometry?.dispose();
                //     //     this.scene.hoverLine.material?.dispose();
                //     //     this.scene.hoverLine = undefined;
                //     // }
                // }
        }

        this.__registerContainerEventListener('mousemove', (e) => { onMouseMove.call(this, e, false); });
        // FIXME click-to-lock logic was mixed into the logic of mousemove
        this.__registerContainerEventListener('click', (e) => { onMouseMove.call(this, e, true); });
        this.vueApp.$watch('curIndex', this.__updateLastHoveredIndex.bind(this));
    }

    __addDoubleClickLockingControl() {
        function onDoubleClick(event) {
            // Raycasting to find the intersected point
            let rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(mouse, this.camera);
            let intersects = raycaster.intersectObject(this.pointsMesh);

            if (intersects.length > 0 && checkVisibility(this.pointsMesh.geometry.attributes.alpha.array, intersects[0].index)) {
                if (this.vueApp.selectedIndex != null) {
                    this.pointsMesh.geometry.attributes.size.array[this.vueApp.selectedIndex] = NORMAL_SIZE;
                    this.pointsMesh.geometry.attributes.size.needsUpdate = true;
                }
                // Get the index and position of the double-clicked point
                let intersect = intersects[0];
                if (this.vueApp.selectedIndex.includes(intersect.index)) {
                    return;
                }
                console.log("this.vueApp.selectedIndex", this.vueApp.selectedIndex);
                this.vueApp.selectedIndex.push(intersect.index);
                console.log("this.vueApp.selectedIndex after push", this.vueApp.selectedIndex);

                let camera = this.camera;
                let canvas = this.renderer.domElement;
                console.log("this.vueApp.selectedInde", this.vueApp.selectedPointPos);

                let vector = intersect.point.clone().project(camera);


                vector.x = Math.round((vector.x * 0.5 + 0.5) * canvas.clientWidth);
                vector.y = - Math.round((vector.y * 0.5 - 0.5) * canvas.clientHeight);

                let rect = canvas.getBoundingClientRect();
                vector.x += rect.left;
                vector.y += rect.top;

                this.vueApp.selectedPointPos.push({ 'x': vector.x, 'y': vector.y });
                console.log("this.vueApp.selectedIndex after push", this.vueApp.selectedPointPos);
                //   this.pointsMesh.geometry.attributes.size.array[this.vueApp.selectedIndex] = HOVER_SIZE;
                this.vueApp.selectedIndex.forEach(index => {
                    this.pointsMesh.geometry.getAttribute('size').array[index] = SELECTED_SIZE;
                });
                this.pointsMesh.geometry.getAttribute('size').needsUpdate = true;
            }
        }

        this.__registerContainerEventListener('dblclick', onDoubleClick);
    }

    __addFilterTestTrain() {
        let specifiedShowTesting = makeSpecifiedVariableName('showTesting', '');
        let specifiedShowTraining = makeSpecifiedVariableName('showTraining', '');
        let specifiedPredictionFlipIndices = makeSpecifiedVariableName('predictionFlipIndices', '');
        // let specifiedOriginalSettings = makeSpecifiedVariableName('originalSettings', '')

        const updateCurrentDisplay = () => {
            console.log("currDisplay");
            let specifiedTrainIndex = makeSpecifiedVariableName('train_index', '');
            let specifiedTestIndex = makeSpecifiedVariableName('test_index', '');

            this.pointsMesh.geometry.attributes.alpha.array = updateShowingIndices(this.pointsMesh.geometry.attributes.alpha.array, this.vueApp[specifiedShowTraining], this.vueApp[specifiedTrainIndex], this.vueApp[specifiedPredictionFlipIndices]);
            this.pointsMesh.geometry.attributes.alpha.array = updateShowingIndices(this.pointsMesh.geometry.attributes.alpha.array, this.vueApp[specifiedShowTesting], this.vueApp[specifiedTestIndex], this.vueApp[specifiedPredictionFlipIndices]);

            // update position z index to allow currDisplay indices show above
            for (let i = 0; i < this.pointsMesh.geometry.attributes.alpha.array.length; i++) {
                let zIndex = i * 3 + 2;
                this.pointsMesh.geometry.attributes.position.array[zIndex] = this.pointsMesh.geometry.attributes.alpha.array[i] === 1 ? 0 : -1;
            }


            this.pointsMesh.geometry.attributes.position.needsUpdate = true;
            this.pointsMesh.geometry.attributes.alpha.needsUpdate = true;
        }

        this.vueApp.$watch(specifiedShowTesting, updateCurrentDisplay);
        this.vueApp.$watch(specifiedShowTraining, updateCurrentDisplay);
        this.vueApp.$watch(specifiedPredictionFlipIndices, updateCurrentDisplay);
        // this.vueApp.$watch(specifiedOriginalSettings, resetToOriginalColorSize, { deep: true });
    }

    __addHighlight() {
        const resetToOriginalColorSize = () => {
            let specifiedSelectedIndex = makeSpecifiedVariableName('selectedIndex', '');
            this.pointsMesh.geometry.getAttribute('size').array.set(this.vueApp.originalSettings.originalSizes);
            this.pointsMesh.geometry.getAttribute('color').array.set(this.vueApp.originalSettings.originalColors);
            // not reset selectedIndex
            if (this.vueApp[specifiedSelectedIndex]) {
                this.pointsMesh.geometry.getAttribute('size').array[this.vueApp[specifiedSelectedIndex]] = HOVER_SIZE;
            }

            // Mark as needing update
            this.pointsMesh.geometry.getAttribute('size').needsUpdate = true;
            this.pointsMesh.geometry.getAttribute('color').needsUpdate = true;
            // console.log("resetColor", this.vueApp.originalSettings.originalColors)
        }

        const updateColorSizeForHighlights = (visualizationError) => {
            visualizationError.forEach(index => {
                this.pointsMesh.geometry.getAttribute('size').array[index] = HOVER_SIZE;
            });
            this.pointsMesh.geometry.getAttribute('size').needsUpdate = true;

            // yellow indices are triggered by right selected index
            visualizationError.forEach(index => {
                this.pointsMesh.geometry.getAttribute('color').array[index * 3] = GRAY[0]; // R
                this.pointsMesh.geometry.getAttribute('color').array[index * 3 + 1] = GRAY[1]; // G
                this.pointsMesh.geometry.getAttribute('color').array[index * 3 + 2] = GRAY[2]; // B
            });

            this.pointsMesh.geometry.getAttribute('color').needsUpdate = true;
        }

        const updateHighlights = () => {
            console.log("updateHihglight");
            let visualizationError = this.vueApp[specifiedHighlightAttributes].visualizationError;
            if (visualizationError == null) {
                visualizationError = [];
            } else {
                visualizationError = Array.from(visualizationError);
            }
            resetToOriginalColorSize();

            updateColorSizeForHighlights(visualizationError);
            visualizationError = [];
        }

        // In the Vue instance where you want to observe changes
        let specifiedHighlightAttributes = makeSpecifiedVariableName('highlightAttributes', '');
        this.vueApp.$watch(specifiedHighlightAttributes, updateHighlights, {
            deep: true // Use this if specifiedHighlightAttributes is an object to detect nested changes
        });
    }

    __moveCameraWithinRange(camera, newPosX, newPosY) {
        if (newPosX === undefined || newPosY === undefined) {
            newPosX = camera.position.x;
            newPosY = camera.position.y;
        }

        const currentZoom = camera.zoom;

        // left bound: minX <= x - w / 2 / scale,
        // right bound: maxX >= x + w / 2 / scale
        // so does y

        const minX = this.boundary.x_min + (camera.right - camera.left) / 2 / currentZoom;
        const maxX = this.boundary.x_max - (camera.right - camera.left) / 2 / currentZoom;
        const minY = this.boundary.y_min + (camera.top - camera.bottom) / 2 / currentZoom;
        const maxY = this.boundary.y_max - (camera.top - camera.bottom) / 2 / currentZoom;

        newPosX = Math.max(minX, Math.min(newPosX, maxX));
        newPosY = Math.max(minY, Math.min(newPosY, maxY));

        // update camera position
        camera.position.x = newPosX;
        camera.position.y = newPosY;
    }
}

function drawCanvas(res) {
    // if (window.vueApp.plotCanvas) {
        //     window.vueApp.plotCanvas.disposeResources();
        // }
    if (window.vueApp.plotCanvas) {
        window.vueApp.plotCanvas.plotDataPoints(res, true);
    } else {
        let container = document.getElementById("container");
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }
        const plotCanvas = new PlotCanvas(window.vueApp);
        plotCanvas.bindTo(container);
        plotCanvas.plotDataPoints(res);
        plotCanvas.render();

        window.vueApp.plotCanvas = plotCanvas;
    }

    window.vueApp.isCanvasLoading = false;
}

function updateSizes() {
    // const nn = []; // 创建一个空的 sizes 列表
    window.vueApp.nnIndices.forEach((item, index) => {
        window.vueApp.pointsMesh.geometry.attributes.size.array[item] = NORMAL_SIZE;
    });
    window.vueApp.pointsMesh.geometry.attributes.size.needsUpdate = true;
    window.vueApp.nnIndices = [];
    Object.values(window.vueApp.query_result).forEach(item => {
        if (typeof item === 'object' && item !== null) {
            window.vueApp.nnIndices.push(item.id);
            window.vueApp.nnIndices.push(item.id);
        }
    });
    console.log(window.vueApp.nnIndices);
    window.vueApp.nnIndices.forEach((item, index) => {
        window.vueApp.pointsMesh.geometry.attributes.size.array[item] = HOVER_SIZE;
    });
    window.vueApp.pointsMesh.geometry.attributes.size.needsUpdate = true;
    resultContainer = document.getElementById("resultContainer");
    resultContainer.setAttribute("style", "display:block;");
}

function clear() {
    window.vueApp.nnIndices.forEach((item, index) => {
        window.vueApp.pointsMesh.geometry.attributes.size.array[item] = NORMAL_SIZE;
    });
    window.vueApp.pointsMesh.geometry.attributes.size.needsUpdate = true;
    window.vueApp.nnIndices = [];
    resultContainer = document.getElementById("resultContainer");
    resultContainer.setAttribute("style", "display:none;");
}

function show_query_text() {
    resultContainer = document.getElementById("resultContainer");
    resultContainer.setAttribute("style", "display:block;");
}

function labelColor() {
    const labels = window.vueApp.label_name_dict;
    const colors = window.vueApp.color_list;

    const tableBody = document.querySelector('#labelColor tbody');
    tableBody.innerHTML = '';
    const hexToRgbArray = (hex) => {
        hex = hex.replace(/^#/, '');
        let bigint = parseInt(hex, 16);
        let r = (bigint >> 16) & 255;
        let g = (bigint >> 8) & 255;
        let b = bigint & 255;
        return [r, g, b];
    };

    function changeLabelColor(label2Change, newColor) {
        const pointLabels = window.vueApp.label_list;
        for (let i = 0; i < pointLabels.length; i++) {
            if (pointLabels[i] === label2Change) {
                window.vueApp.res.label_color_list[i][0] = newColor[0];
                window.vueApp.res.label_color_list[i][1] = newColor[1];
                window.vueApp.res.label_color_list[i][2] = newColor[2];
            }
        }
        drawCanvas(window.vueApp.res)
    }

    Object.keys(labels).forEach((key, index) => {
        const row = document.createElement('tr');

        // 创建标签名单元格
        const labelCell = document.createElement('td');
        labelCell.textContent = labels[key];
        row.appendChild(labelCell);

        // 创建颜色单元格
        const colorCell = document.createElement('td');
        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.value = `#${colors[index].map(c => c.toString(16).padStart(2, '0')).join('')}`;
        colorInput.addEventListener('input', (event) => {
                const newColor = event.target.value;
                changeLabelColor(key, hexToRgbArray(newColor));
            }
        );
        colorCell.appendChild(colorInput);
        row.appendChild(colorCell);
        // 将行添加到表格中
        tableBody.appendChild(row);
    });
}
