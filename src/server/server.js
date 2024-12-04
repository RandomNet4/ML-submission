const Hapi = require("@hapi/hapi");
require("dotenv").config();
const loadModel = require("../services/loadModel");
const routes = require("./routes");
const InputError = require("../exceptions/InputError");

const init = async () => {
  const server = Hapi.server({
    port: 3000,
    host: "0.0.0.0",
    routes: {
      cors: {
        origin: ["*"],
      },
    },
  });

  const model = await loadModel();
  server.app.model = model;

  server.route(routes);

  server.ext("onPreResponse", (request, h) => {
    const { response } = request;
    if (response instanceof InputError) {
      const newResponse = h.response({
        status: "fail!",
        message: `Terdapat kesalahan dalam melakukan prediksi`,
      });
      newResponse.code(response.statusCode);
      return newResponse;
    }
    if (response.isBoom) {
      if (response.output.statusCode === 413) {
        const customResponse = h
          .response({
            status: "fail!",
            message:
              "Payload content length greater than maximum allowed: 1000000",
          })
          .code(413);
        return customResponse;
      }

      const newResponse = h.response({
        status: "fail!",
        message: response.message,
      });
      newResponse.code(response.output.statusCode);
      return newResponse;
    }
    return h.continue;
  });

  await server.start();
  console.log("Server running on %s", server.info.uri);
};

init();
