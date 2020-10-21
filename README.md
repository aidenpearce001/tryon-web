# Handpose demo

## Contents

This demo shows how to use the handpose model to detect hands in a video stream.

## Setup

Via npm:

Using yarn:
```sh
$ 
$ yarn add @tensorflow/tfjs-coryarn add @tensorflow-models/handposee, @tensorflow/tfjs-converter
$ yarn add @tensorflow/tfjs-backend-webgl # or @tensorflow/tfjs-backend-wasm
```

Install dependencies and prepare the build directory:

```
yarn
```

To watch files for changes, and launch a dev server:

```
yarn watch
```

## If you are developing handpose locally, and want to test the changes in the demos


Install dependencies:
```
yarn
```

Publish handpose locally:
```
yarn build && yarn yalc publish
```

Cd into the demos and install dependencies:

```
yarn
```

Link the local handpose to the demos:
```
yarn yalc link @tensorflow-models/handpose
```

Start the dev demo server:
```
yarn watch
```

To get future updates from the handpose source code:
```sh
# cd up into the handpose directory
cd ../
yarn build && yarn yalc push
```
