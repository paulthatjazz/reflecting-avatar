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

import * as facemesh from '@tensorflow-models/facemesh';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// TODO(annxingyuan): read version from tfjsWasm directly once
// https://github.com/tensorflow/tfjs/pull/2819 is merged.
import { version } from '@tensorflow/tfjs-backend-wasm/dist/version';

import { TRIANGULATION } from './triangulation';

tfjsWasm.setWasmPath(
	`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${version}/dist/tfjs-backend-wasm.wasm`
);

function isMobile() {
	const isAndroid = /Android/i.test(navigator.userAgent);
	const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
	return isAndroid || isiOS;
}

function drawPath(ctx, points, closePath) {
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

let model,
	ctx,
	videoWidth,
	videoHeight,
	video,
	canvas,
	scatterGLHasInitialized = false,
	scatterGL;

// visual && point config
const config = {
	key_point_color: 'red',
	key_point_color_mouth: 'yellow',
	point_color: 'black',
	point_radius: 1.5,
	line_color: 'hotpink',
	line_width: 2.5,
	show_labels: false,
	far_left: 454,
	far_right: 234,
	eye_level: 6,
	mouth_level: 0,
	middle: 4
};

const mouth_tm = 13;
//const mouth = [ 78, 14, 311, 402, 81, 178, 308 ];

const mouth = [ 14 ];

let faceroll = {
	abs_x: 0,
	abs_y: 0, //face position (X,Y)
	tilt: 0, //tilt angle
	roll_x: 0,
	roll_y: 0 //roll angle (X,Y)
};

const VIDEO_SIZE = 400;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
const renderPointcloud = mobile === false;
const state = {
	backend: 'wasm',
	maxFaces: 1,
	triangulateMesh: false
};

if (renderPointcloud) {
	state.renderPointcloud = true;
}

async function setupCamera() {
	video = document.getElementById('video');

	const stream = await navigator.mediaDevices.getUserMedia({
		audio: false,
		video: {
			facingMode: 'user',
			// Only setting the video to a specified size in order to accommodate a
			// point cloud, so on mobile devices accept the default size.
			width: mobile ? undefined : VIDEO_SIZE,
			height: mobile ? undefined : VIDEO_SIZE
		}
	});
	video.srcObject = stream;

	return new Promise((resolve) => {
		video.onloadedmetadata = () => {
			resolve(video);
		};
	});
}

async function renderPrediction() {
	const predictions = await model.estimateFaces(video);
	ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);

	if (predictions.length > 0) {
		predictions.forEach((prediction) => {
			const keypoints = prediction.scaledMesh;

			if (state.triangulateMesh) {
				for (let i = 0; i < TRIANGULATION.length / 3; i++) {
					const points = [ TRIANGULATION[i * 3], TRIANGULATION[i * 3 + 1], TRIANGULATION[i * 3 + 2] ].map(
						(index) => keypoints[index]
					);

					drawPath(ctx, points, true);
				}
			} else {
				//normalise around the middle point, for face roll angle.

				let x_dis = Math.floor(keypoints[config.far_left][0]) - Math.floor(keypoints[config.far_right][0]);
				let normalised_mid_x =
					Math.floor(keypoints[config.middle][0]) - Math.floor(keypoints[config.far_right][0]);

				let y_dis = Math.floor(keypoints[config.mouth_level][1]) - Math.floor(keypoints[config.eye_level][1]);
				let normalised_mid_y =
					Math.floor(keypoints[config.middle][1]) - Math.floor(keypoints[config.eye_level][1]);

				//percentage representing faceroll angle X-Axis

				faceroll.roll_x = normalised_mid_x / (x_dis / 100);

				//percentage representing faceroll angle Y-Axis

				faceroll.roll_y = normalised_mid_y / (y_dis / 100);

				faceroll.tilt = Math.floor(keypoints[config.far_left][1]) - Math.floor(keypoints[config.far_right][1]);

				faceroll.abs_x = keypoints[config.middle][0];
				faceroll.abs_y = keypoints[config.middle][1];

				//top middle mouth will always = [0, 0]
				let mouth_relative = [];
				//normalise mouth values relative to top middle mouth
				for (let x = 0; x < mouth.length; x++) {
					mouth_relative.push([
						keypoints[x][0] - keypoints[mouth_tm][0],
						keypoints[x][1] - keypoints[mouth_tm][1],
						keypoints[x][2] - keypoints[mouth_tm][2]
					]);
				}

				mouth_ex = mouth_relative;

				document.querySelector('#debug').innerHTML =
					'<h2>Face Roll Angle Values</h2><br>Middle (Abs) : ' +
					Math.floor(faceroll.abs_x) +
					', ' +
					Math.floor(faceroll.abs_y) +
					'<br> Distance(X): ' +
					x_dis +
					'<br> X roll: ' +
					normalised_mid_x +
					' (' +
					Math.floor(faceroll.roll_x) +
					'%)' +
					'<br> Distance(Y): ' +
					y_dis +
					'<br> Y roll: ' +
					normalised_mid_y +
					' (' +
					Math.floor(faceroll.roll_y) +
					'%)' +
					'<br> Tilt:' +
					faceroll.tilt;

				for (let i = 0; i < keypoints.length; i++) {
					const x = keypoints[i][0];
					const y = keypoints[i][1];

					if (
						i == config.far_left ||
						i == config.far_right ||
						i == config.middle ||
						i == config.eye_level ||
						i == config.mouth_level
					) {
						ctx.fillStyle = config.key_point_color;
					} else if (mouth.includes(i)) {
						ctx.fillStyle = config.key_point_color_mouth;
					} else if (i == mouth_tm) {
						ctx.fillStyle = 'limegreen';
					} else {
						ctx.fillStyle = config.point_color;
					}

					ctx.beginPath();
					ctx.arc(x, y, config.point_radius, 0, 2 * Math.PI);
					if (config.show_labels) {
						ctx.fillText(i, x, y);
					}
					ctx.fill();
				}

				// line between FL & FR
				ctx.beginPath();
				ctx.moveTo(keypoints[config.far_right][0], keypoints[config.far_right][1]);
				ctx.lineTo(keypoints[config.far_left][0], keypoints[config.far_left][1]);
				ctx.stroke();
			}
		});

		if (renderPointcloud && state.renderPointcloud && scatterGL != null) {
			const pointsData = predictions.map((prediction) => {
				let scaledMesh = prediction.scaledMesh;
				return scaledMesh.map((point) => [ -point[0], -point[1], -point[2] ]);
			});

			let flattenedPointsData = [];
			for (let i = 0; i < pointsData.length; i++) {
				flattenedPointsData = flattenedPointsData.concat(pointsData[i]);
			}
			const dataset = new ScatterGL.Dataset(flattenedPointsData);

			if (!scatterGLHasInitialized) {
				scatterGL.render(dataset);
			} else {
				scatterGL.updateDataset(dataset);
			}
			scatterGLHasInitialized = true;
		}
	}

	frav = faceroll; // update front end with face roll angle values.

	requestAnimationFrame(renderPrediction);
}

async function main() {
	await tf.setBackend(state.backend);

	await setupCamera();
	video.play();
	videoWidth = video.videoWidth;
	videoHeight = video.videoHeight;
	video.width = videoWidth;
	video.height = videoHeight;

	canvas = document.getElementById('output');
	canvas.width = videoWidth;
	canvas.height = videoHeight;
	const canvasContainer = document.querySelector('.canvas-wrapper');
	canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

	ctx = canvas.getContext('2d');
	ctx.translate(canvas.width, 0);
	ctx.scale(-1, 1);

	ctx.fillStyle = config.point_color;
	ctx.strokeStyle = config.line_color;
	ctx.lineWidth = config.line_width;

	model = await facemesh.load({ maxFaces: state.maxFaces });
	renderPrediction();
}

main();
