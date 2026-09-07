FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3001

COPY package.json package-lock.json* ./

RUN npm install

COPY . .

RUN npm run build

EXPOSE 3001

CMD ["npx", "next", "start", "-p", "3001"]
