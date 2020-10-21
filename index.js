/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-webgl';

tfjsWasm.setWasmPath(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/tfjs-backend-wasm.wasm`);

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

let videoWidth, videoHeight, rafID, ctx, canvas, ANCHOR_POINTS,
    scatterGLHasInitialized = false, scatterGL, fingerLookupIndices = {
      thumb: [0, 1, 2, 3, 4],
      // indexFinger: [0, 5, 6, 7, 8],
      // middleFinger: [0, 9, 10, 11, 12],
      ringFinger: [0, 13, 14, 15, 16],
      // pinky: [0, 17, 18, 19, 20]
    };  // for rendering each finger as a polyline

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.

// set to true if need to render keypoint cloud
const renderPointcloud = false;

const state = {
  backend: 'webgl'
};

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

let timestep = 0;
let globalKeypoints = [];

if (renderPointcloud) {
  state.renderPointcloud = true;
}

function setupDatGui() {
  const gui = new dat.GUI();
  gui.add(state, 'backend', ['webgl', 'wasm'])
      .onChange(async backend => {
        window.cancelAnimationFrame(rafID);
        await tf.setBackend(backend);
        landmarksRealTime(video);
      });

  if (renderPointcloud) {
    gui.add(state, 'renderPointcloud').onChange(render => {
      document.querySelector('#scatter-gl-container').style.display =
          render ? 'inline-block' : 'none';
    });
  }
}

import image from "./public/obj.jpg";
function drawPoint(y, x, r) {

  ctx.beginPath();
  // ctx.arc(x, y, r, 0, 2 * Math.PI);
  let img_component = document.createElement("img");
  img_component.src = image;
  img_component.setAttribute("height",10);
  img_component.setAttribute("width",10);
  ctx.drawImage(img_component, x, y, 30, 30);
  ctx.fill();
}

function drawKeypoints(keypoints) {
  timestep += 1;
  let keypointsArray = keypoints;
  if (timestep % 5 !== 0) {
    keypointsArray = globalKeypoints;
  } else {
    globalKeypoints = keypointsArray;
  }


  keypointsArray = keypointsArray.slice(13,16);
  
  console.log("KEYPOINTS:",keypointsArray);

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(x, y, 3);
  }

  // const fingers = Object.keys(fingerLookupIndices);
  // for (let i = 0; i < fingers.length; i++) {
  //   const finger = fingers[i];
  //   const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
  //   drawPath(points, false);
  // }
}

function drawPath(points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model;

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_WIDTH,
      height: mobile ? undefined : VIDEO_HEIGHT
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}

async function main() {
  await tf.setBackend(state.backend);
  model = await handpose.load();
  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = e.message;
    info.style.display = 'block';
    throw e;
  }

  setupDatGui();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = 'red';
  ctx.fillStyle = 'red';

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  // These anchor points allow the hand pointcloud to resize according to its
  // position in the input.
  ANCHOR_POINTS = [
    [0, 0, 0], [0, -VIDEO_HEIGHT, 0], [-VIDEO_WIDTH, 0, 0],
    [-VIDEO_WIDTH, -VIDEO_HEIGHT, 0]
  ];

  if (renderPointcloud) {
    document.querySelector('#scatter-gl-container').style =
        `width: ${VIDEO_WIDTH}px; height: ${VIDEO_HEIGHT}px;`;

    scatterGL = new ScatterGL(
        document.querySelector('#scatter-gl-container'),
        {'rotateOnStart': false, 'selectEnabled': false});
  }

  landmarksRealTime(video);
}

const landmarksRealTime = async (video) => {
  async function frameLandmarks() {
    stats.begin();
    ctx.drawImage(
        video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
        canvas.height);
    const predictions = await model.estimateHands(video);
    if (predictions.length > 0) {
      const result = predictions[0].landmarks;
      drawKeypoints(result, predictions[0].annotations);

      if (renderPointcloud === true && scatterGL != null) {
        const pointsData = result.map(point => {
          return [-point[0], -point[1], -point[2]];
        });

        const dataset =
            new ScatterGL.Dataset([...pointsData, ...ANCHOR_POINTS]);

        if (!scatterGLHasInitialized) {
          scatterGL.render(dataset);

          const fingers = Object.keys(fingerLookupIndices);

          scatterGL.setSequences(
              fingers.map(finger => ({indices: fingerLookupIndices[finger]})));
          scatterGL.setPointColorer((index) => {
            if (index < pointsData.length) {
              return 'steelblue';
            }
            return 'white';  // Hide.
          });
        } else {
          scatterGL.updateDataset(dataset);
        }
        scatterGLHasInitialized = true;
      }
    }
    stats.end();
    rafID = requestAnimationFrame(frameLandmarks);
  };

  frameLandmarks();
};

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

main();
