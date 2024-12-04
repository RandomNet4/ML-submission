const { postBinaryHandler, getHistoriesHandler } = require("../server/handler");

const routes = [
  {
    path: "/predict",
    method: "POST",
    handler: postBinaryHandler,
    options: {
      payload: {
        allow: "multipart/form-data",
        multipart: true,
        maxBytes: 1048576,
      },
    },
  },
  {
    path: "/predict/histories",
    method: "GET",
    handler: getHistoriesHandler,
  },
];

module.exports = routes;
