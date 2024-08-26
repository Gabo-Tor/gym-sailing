import os

import numpy as np
import pygame


class Renderer:
    WATER_COLOR = (38, 102, 138)
    BOAT_COLOR = (220, 245, 230)
    TARGET_COLOR = (255, 150, 0)
    INFO_COLOR = (102, 160, 198)
    FONT_SIZE = 20

    def __init__(
        self,
        boat_length=4.2,
        boat_beam=1.4,
        target_radius=1.3 / 2,
        course_size=60,
    ):
        self.boat_length = boat_length
        self.boat_beam = boat_beam
        self.target_rad = 0.3 * target_radius  # TODO: get a better fix for this 0.3
        self.course_size = course_size

        self.screen_width = 680

        self.scale = self.screen_width / (self.course_size)
        self.screen_height = int(self.scale * (self.course_size))

        pygame.font.init()
        self.normal_font = pygame.font.SysFont("monospace", 20)

        boatwidth = self.boat_beam * self.scale
        boatlength = self.boat_length * self.scale

        path = os.path.dirname(os.path.abspath(__file__))

        self.boat_img = pygame.image.load(os.path.join(path, "assets/laser.png"))

        self.boat_img = pygame.transform.scale(
            self.boat_img, (boatwidth * 20, boatlength * 20)
        )
        self.sail_img = pygame.image.load(os.path.join(path, "assets/sail.png"))
        self.sail_img = pygame.transform.scale(
            self.sail_img, (boatwidth * 20, 1.505 * boatlength * 20)
        )
        self.trail = []

        self.window = None
        self.clock = None

    def _render_frame(self, boats, target, stepnum, reward, render_mode, fps):
        if self.window is None and render_mode in ["human", "rgb_array"]:
            # FIXME: self.window should be used only in human mode,
            # in rgb mode there is no need to create a window, only the surface
            # the surface should be separated from the window
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Boat Environment")

        if self.clock is None and render_mode == "human":
            self.clock = pygame.time.Clock()

        self.draw_water()
        self.draw_target(target)

        for n, boat in enumerate(boats):
            if n == 0:  # for the first boat only, draw the trail
                self.draw_trail(boat)

            boat_heading = boat[2]
            boat_pos = (boat[0], boat[1])
            if len(boat) > 3:
                rudder = -0.5 * boat[3]
            else:
                rudder = 0

            if len(boat) > 4:
                boat_type = boat[4]
                assert boat_type in ["sailboat", "motorboat", "iceboat"]
            else:
                boat_type = "sailboat"

            self.draw_boat(
                boat_pos, boat_heading, rudder, self.boat_color(n), boat_type
            )

        self.window.blit(pygame.transform.flip(self.window, False, True), (0, 0))
        self.draw_info(stepnum, reward)

        if render_mode == "human":
            pygame.display.flip()
            pygame.event.pump()
            pygame.display.update()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            self.clock.tick(fps)

        elif render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window).transpose(1, 0, 2)

    def draw_trail(self, boat):
        self.trail.append((boat[0], boat[1]))
        if len(self.trail) > 300:
            self.trail.pop(0)
        for i in range(len(self.trail) - 1):
            pygame.draw.aaline(
                self.window,
                self.BOAT_COLOR,
                (
                    int(self.scale * self.trail[i][0]),
                    int(self.scale * self.trail[i][1]),
                ),
                (
                    int(self.scale * self.trail[i + 1][0]),
                    int(self.scale * self.trail[i + 1][1]),
                ),
            )

    def boat_color(self, n):
        if n == 0:
            return self.BOAT_COLOR
        color = (
            130 + (19016231 * n) % 77,
            130 + (44162351 * n) % 72,
            130 + (79114639 * n) % 75,
        )

        return color

    def draw_water(self):
        self.window.fill(Renderer.WATER_COLOR)

    def draw_info(self, stepnum, reward):
        info_label = self.normal_font.render(
            f"step:{stepnum:4} reward:{reward:+4.2f}",
            True,
            Renderer.INFO_COLOR,
        )
        self.window.blit(info_label, (7, self.screen_height - 25))

    def draw_boat(
        self, boat_pos, boat_heading, rudder, color=None, boat_type="sailboat"
    ):
        delta = (
            -np.array([np.sin(boat_heading), np.cos(-boat_heading)])
            * 0.24
            * self.boat_length
            * self.scale
        )
        self.draw_hull(boat_pos, boat_heading, color)
        if boat_type == "sailboat":
            self.draw_sail(boat_pos, boat_heading, delta)
        self.draw_rudder(boat_pos, boat_heading, rudder, delta)

    def draw_hull(self, boat_pos, boat_heading, color):
        boat_img = pygame.transform.rotozoom(
            self.boat_img, np.degrees(-boat_heading), 0.05
        )

        boat_rect = boat_img.get_rect()
        boat_rect.center = (
            int(boat_pos[0] * self.scale),
            int(boat_pos[1] * self.scale),
        )

        if color is not None:
            boat_img.fill(color, special_flags=pygame.BLEND_RGB_MULT)

        self.window.blit(boat_img, boat_rect)

    def draw_sail(self, boat_pos, boat_heading, delta):
        norm_heading = norm(boat_heading)
        if abs(norm_heading) < 0.5:  # replace by line
            pygame.draw.aaline(
                self.window,
                (0, 0, 0),
                (
                    boat_pos[0] * self.scale + delta[0],
                    boat_pos[1] * self.scale - delta[1],
                ),
                (
                    int((boat_pos[0]) * self.scale),
                    int((boat_pos[1] - self.boat_length * 0.5) * self.scale),
                ),
            )

        else:
            if norm_heading > 0:
                sail_img = pygame.transform.flip(self.sail_img, True, False)
                sail_img = pygame.transform.rotozoom(
                    sail_img, np.degrees(-0.45 * (norm_heading + 0.92)), 0.05
                )

            else:
                sail_img = pygame.transform.rotozoom(
                    self.sail_img.copy(),
                    np.degrees(-0.45 * (norm_heading - 0.92)),
                    0.05,
                )

            pos = (
                boat_pos[0] * self.scale + delta[0],
                boat_pos[1] * self.scale - delta[1],
            )
            self.window.blit(sail_img, sail_img.get_rect(center=pos))

    def draw_rudder(self, boat_pos, boat_heading, rudder, delta):
        rudder_angle = boat_heading - rudder
        rudder_length = 0.4 * self.boat_beam * self.scale
        pygame.draw.line(
            self.window,
            (0, 0, 0),
            (
                boat_pos[0] * self.scale - 2 * delta[0],
                boat_pos[1] * self.scale + 2 * delta[1],
            ),
            (
                boat_pos[0] * self.scale
                - 2 * delta[0]
                + rudder_length * np.sin(rudder_angle),
                boat_pos[1] * self.scale
                + 2 * delta[1]
                - rudder_length * np.cos(rudder_angle),
            ),
            width=2,
        )

    def draw_target(self, target):
        layline_length = 1000
        layline_angle = 40

        layline1 = [
            (target[0] + layline_length * np.cos(np.radians(layline_angle))),
            (target[1] - layline_length * np.sin(np.radians(layline_angle))),
        ]

        layline2 = [
            (target[0] - layline_length * np.cos(np.radians(layline_angle))),
            (target[1] - layline_length * np.sin(np.radians(layline_angle))),
        ]

        pygame.draw.aaline(
            self.window,
            Renderer.INFO_COLOR,
            (int(self.scale * target[0]), int(self.scale * target[1])),
            (int(self.scale * layline1[0]), int(self.scale * layline1[1])),
        )

        pygame.draw.aaline(
            self.window,
            Renderer.INFO_COLOR,
            (int(self.scale * target[0]), int(self.scale * target[1])),
            (int(self.scale * layline2[0]), int(self.scale * layline2[1])),
        )

        # Draw the target
        target_radius = int(self.target_rad * self.scale)
        pygame.draw.circle(
            self.window,
            Renderer.TARGET_COLOR,
            (int(self.scale * target[0]), int(self.scale * target[1])),
            target_radius,
        )

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            quit()


def norm(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
    return (angle + np.pi) % (2 * np.pi) - np.pi
